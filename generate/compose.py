#!/usr/bin/env python3
"""
Synthetic data augmentation: paste foreground objects from selected frames
onto inpainted backgrounds using SAM segmentation.

Dataset layout (input):
    dataset_dir/
    ├── source/
    │   ├── A_B.ext
    │   └── {shot}-shot_new.json
    ├── background/
    │   └── {matched_stem}_.jpg
    └── frame/
        └── A_B/
            ├── frame_0.jpg
            └── ...

Output layout:
    output_dir/
    ├── train_{shot}shot/
    │   └── generate_{...}.jpg          # generated images
    └── annotations/
        └── {shot}_shot.json            # updated COCO annotations
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_bbox(json_data: dict, filename: str) -> list[float]:
    """Get bbox [x, y, w, h] for the given filename."""
    for img in json_data["images"]:
        if img["file_name"] == filename:
            for anno in json_data["annotations"]:
                if anno["image_id"] == img["id"]:
                    return anno["bbox"]
    raise KeyError(f"No bbox found for {filename}")


def get_match_info(json_data: dict, filename: str):
    """Return (matched_image_dict, match_index, source_image_id) for a source filename."""
    id_map = json_data["id_map"]
    images = json_data["images"]
    for idx, img in enumerate(images):
        if img["file_name"] == filename:
            src_id = img["id"]
            match_id = id_map[str(src_id)]
            for match_idx, match_img in enumerate(images):
                if match_img["id"] == match_id:
                    return match_img, match_idx, src_id
    raise KeyError(f"Image {filename} not found or no match")


def get_anno_count(id_map: dict, image_id: int) -> int:
    """Count how many annotations share the same matched image."""
    match_id = id_map[str(image_id)]
    return sum(1 for v in id_map.values() if v == match_id)


def crop_white_border(img: np.ndarray, threshold: int = 245) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.nonzero(gray < threshold)
    if not len(coords[0]):
        return img
    y1, y2 = coords[0].min(), coords[0].max() + 1
    x1, x2 = coords[1].min(), coords[1].max() + 1
    return img[y1:y2, x1:x2]


def segment_foreground(predictor: SamPredictor, image: np.ndarray, padding: int = 150) -> np.ndarray:
    """Run SAM on padded image and return foreground mask (original size)."""
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(
        image, padding, padding, padding, padding,
        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    predictor.set_image(padded)
    box = np.array([padding, padding, w + padding, h + padding])
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]
    return mask[padding:padding + h, padding:padding + w]


def paste_foreground(
    background: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    bbox: list[float],
    scale_jitter: float = 0.0,
    random_place: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Scale, optionally randomly place, and blend source onto background."""
    bg = background.copy()
    src = crop_white_border(source)
    mask_src = mask.copy()

    x, y, w, h = bbox
    ratio = (1 + scale_jitter) * min(w / src.shape[1], h / src.shape[0])
    new_w = round(src.shape[1] * ratio)
    new_h = round(src.shape[0] * ratio)
    src = cv2.resize(src, (new_w, new_h))
    mask_src = cv2.resize(mask_src.astype(np.uint8), (new_w, new_h)).astype(bool)

    if random_place:
        cx = np.random.randint(new_w // 2, bg.shape[1] - new_w // 2)
        cy = np.random.randint(new_h // 2, bg.shape[0] - new_h // 2)
    else:
        cx = int(x + w / 2)
        cy = int(y + h / 2)

    left = cx - new_w // 2
    top = cy - new_h // 2
    right = left + new_w
    bottom = top + new_h

    left = max(0, left)
    top = max(0, top)
    right = min(bg.shape[1], right)
    bottom = min(bg.shape[0], bottom)

    if left >= right or top >= bottom:
        return bg, [left, top, right - left, bottom - top]

    region = bg[top:bottom, left:right]
    src_roi = src[top - (cy - new_h // 2):bottom - (cy - new_h // 2),
                  left - (cx - new_w // 2):right - (cx - new_w // 2)]
    mask_roi = mask_src[top - (cy - new_h // 2):bottom - (cy - new_h // 2),
                        left - (cx - new_w // 2):right - (cx - new_w // 2)]

    region[mask_roi] = [0, 0, 0]
    region[:] = region + src_roi
    bg[top:bottom, left:right] = region

    return bg, [left, top, right - left, bottom - top]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training images.")
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Input dataset directory (contains source/, background/, frame/).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output root directory (will contain train_{shot}shot/ and annotations/).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., clipart1k).")
    parser.add_argument("--shot", type=int, required=True,
                        help="Few-shot number (1,5,10).")
    parser.add_argument("--num_categories", type=int, required=True,
                        help="Number of categories.")
    parser.add_argument("--start_category", type=int, default=1,
                        help="Starting category index (0 for UODD).")
    parser.add_argument("--sam_checkpoint", type=Path, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_variants", type=int, default=3,
                        help="Number of frame variants per image.")
    parser.add_argument("--scale_jitter", type=float, default=0.1,
                        help="Random scale jitter.")
    parser.add_argument("--padding", type=int, default=150,
                        help="Padding for SAM prediction.")
    parser.add_argument("--clear_output", action="store_true",
                        help="Remove existing generate_*.jpg in output train folder before start.")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    # Output structure: output_dir/train_{shot}shot/ and output_dir/annotations/
    train_dir = output_dir / f"train_{args.shot}shot"
    anno_dir = output_dir / "annotations"
    train_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    source_dir = dataset_dir / "source"
    background_dir = dataset_dir / "background"
    frame_dir = dataset_dir / "frame"

    json_new_path = source_dir / f"{args.shot}-shot_new.json"
    json_init_path = source_dir / f"{args.shot}-shot.json"
    if not json_new_path.is_file() or not json_init_path.is_file():
        raise FileNotFoundError("Required JSON annotation files not found in source/.")

    json_data = load_json(json_new_path)
    json_data_init = load_json(json_init_path)

    # Clear existing generated images if requested
    if args.clear_output:
        for f in train_dir.glob("generate_*.jpg"):
            f.unlink()
            print(f"Removed {f.name}")

    # Load SAM
    print("Loading SAM model...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)

    max_id = max(img["id"] for img in json_data_init["images"]) if json_data_init["images"] else 0
    len_anno = len(json_data["annotations"])
    start = args.start_category
    end = start + args.num_categories

    frame_prefix = "frame"

    for category in range(start, end):
        for shot_idx in range(1, args.shot + 1):
            img_name = f"{category}_{shot_idx}.jpg"
            try:
                match_img, match_idx, match_id = get_match_info(json_data, img_name)
            except KeyError:
                continue

            anno_count = get_anno_count(json_data["id_map"], match_id)
            match_stem = Path(match_img["file_name"]).stem
            background_path = background_dir / f"{match_stem}.jpg"
            if not background_path.exists():
                continue
            background = cv2.imread(str(background_path))

            bbox = get_bbox(json_data, img_name)

            for j in range(args.num_variants):
                if j == 0:
                    max_id += 1
                new_id = max_id + len_anno * j

                frame_path = frame_dir / f"{category}_{shot_idx}" / f"{frame_prefix}_{j}.jpg"
                if not frame_path.exists():
                    continue

                source_frame = cv2.imread(str(frame_path))
                if source_frame is None:
                    continue

                mask = segment_foreground(predictor, source_frame, args.padding)
                scale = random.uniform(0, args.scale_jitter) if args.scale_jitter > 0 else 0.0

                composited, new_bbox = paste_foreground(
                    background, source_frame, mask, bbox, scale, random_place=True
                )

                gen_name = f"generate_{category}_{shot_idx}_{j}_{match_idx}_{anno_count}.jpg"
                gen_path = train_dir / gen_name   # save inside train_{shot}shot/
                cv2.imwrite(str(gen_path), composited)

                img_entry = {
                    "file_name": gen_name,
                    "height": composited.shape[0],
                    "width": composited.shape[1],
                    "id": new_id,
                }
                anno_entry = {
                    "id": f"generate_{category}_{shot_idx}_{j}",
                    "iscrowd": 0,
                    "image_id": new_id,
                    "bbox": new_bbox,
                    "category_id": category,
                    "area": mask.sum().item(),
                    "ignore": 0,
                }
                json_data_init["images"].append(img_entry)
                json_data_init["annotations"].append(anno_entry)

                print(f"Generated {gen_name}")

    output_json = anno_dir / f"{args.shot}_shot.json"
    save_json(json_data_init, output_json)
    print(f"Updated annotations saved to {output_json}")


if __name__ == "__main__":
    main()