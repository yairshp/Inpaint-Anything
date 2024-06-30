import os
import sys
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

import utils
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_detection_processor_and_model(device):
    detection_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    detection_model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32"
    ).to(device)
    return detection_processor, detection_model


def get_args(args):
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, nargs="+", required=True)
    parser.add_argument("--text", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", nargs="+", type=str, required=True)
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b", "vit_t"],
        help="The type of sam model to load. Default: 'vit_h",
    )
    parser.add_argument(
        "--dilate_kernel_size",
        type=int,
        default=15,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config",
        type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
        "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt",
        type=str,
        default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    return parser.parse_args(args)


def get_bbox(processor, model, image, texts, device="cuda"):
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=0.1
    )

    if len(results[0]["boxes"]) == 0:
        return None

    boxes, scores = results[0]["boxes"], results[0]["scores"]
    highest_score_bbox = boxes[scores.argmax()].detach().cpu().numpy()
    return highest_score_bbox


def get_center_coords(highest_score_bbox):
    return (
        int((highest_score_bbox[0] + highest_score_bbox[2]) / 2),
        int((highest_score_bbox[1] + highest_score_bbox[3]) / 2),
    )


def preprocess_image(image_path, res):
    image = Image.open(image_path).convert("RGB").resize((res, res))
    return image


def is_object_removed(inpainted_img, bbox, processor, model, text, device="cuda"):
    cropped_img = Image.fromarray(inpainted_img).crop(bbox)
    # cropped_img = inpainted_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    texts = [[text]]
    new_bbox = get_bbox(processor, model, cropped_img, texts, device)
    if new_bbox is None:
        return True
    return False


def get_mse_in_bbox(original_img, inpainted_img, bbox):
    original_img_cropped = original_img.copy()
    original_img_cropped = original_img_cropped[
        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
    ]
    inpainted_img_cropped = inpainted_img.copy()
    inpainted_img_cropped = inpainted_img_cropped[
        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
    ]
    mse = np.mean((original_img_cropped - inpainted_img_cropped) ** 2)
    return mse


def is_background_affected(original_img, inpainted_img, bbox, threshold=0.1):
    # check the mse between the original image and the inpainted image outside the bbox
    original_img_masked = original_img.copy()
    original_img_masked[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])] = 0
    inpainted_img_masked = inpainted_img.copy()
    inpainted_img_masked[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])] = 0
    mse = np.mean((original_img_masked - inpainted_img_masked) ** 2)
    return mse > threshold


def main():
    args = get_args(sys.argv[1:])

    assert len(args.image_path) == len(args.text) == len(args.output_path)

    for image_path, text, output_path in zip(
        args.image_path, args.text, args.output_path
    ):
        # get the center coordinates of the detected object
        image = preprocess_image(image_path, args.res)
        texts = [[text]]
        detection_processor, detection_model = get_detection_processor_and_model(DEVICE)
        bbox = get_bbox(detection_processor, detection_model, image, texts, DEVICE)
        center_coords = get_center_coords(bbox)

        # remove the detected object
        img = utils.load_img_to_array(image_path, res=args.res)
        masks, _, _ = predict_masks_with_sam(
            img,
            [center_coords],
            [1],
            model_type=args.sam_model_type,
            ckpt_p=args.sam_ckpt,
            device=DEVICE,
        )
        masks = masks.astype(np.uint8) * 255

        # dilate mask to avoid unmasked edge effect
        if args.dilate_kernel_size is not None:
            masks = [utils.dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        # inpaint the masked image
        heighest_mse_in_bbox = 0
        best_inpainted_img = None
        for _, mask in enumerate(masks):
            img_inpainted = inpaint_img_with_lama(
                img, mask, args.lama_config, args.lama_ckpt, device=DEVICE
            )
            if is_background_affected(img, img_inpainted, bbox):
                continue
            mse_in_bbox = get_mse_in_bbox(img, img_inpainted, bbox)
            if mse_in_bbox > heighest_mse_in_bbox:
                heighest_mse_in_bbox = mse_in_bbox
                best_inpainted_img = img_inpainted
        if best_inpainted_img is not None:
            utils.save_array_to_img(best_inpainted_img, output_path)
        else:
            print(f"Failed to inpaint the image - {image_path}")


if __name__ == "__main__":
    main()
