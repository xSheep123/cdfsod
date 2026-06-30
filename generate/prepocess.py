"""Remove backgrounds, inpaint matched images, and recenter foregrounds.

Dataset layout:
    dataset_dir/
    ├── source/
    │   ├── images named A_B.ext
    │   └── *shot*.json      # annotations and id_map
    ├── foreground/          # generated RGBA foregrounds
    └── background/          # generated inpainted backgrounds
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rembg

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
IMAGE_NAME_RE = re.compile(
    r"^(?P<category>\d+)_(?P<instance>\d+)\.(?:jpg|jpeg|png|webp|bmp|tiff)$",
    re.IGNORECASE,
)


def parse_image_name(filename: str) -> tuple[str, str]:
    match = IMAGE_NAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(
            f"Invalid filename '{filename}'. Expected A_B.ext, e.g. 3_1.jpg."
        )
    return match.group("category"), match.group("instance")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def find_json(source_dir: Path) -> Path | None:
    """Search for a JSON annotation file inside the source directory.
    Prefers files containing 'shot' in the name.
    """
    shot_files = sorted(
        path for path in source_dir.glob("*.json") if "shot" in path.name.lower()
    )
    if shot_files:
        return shot_files[0]

    for filename in ("annotations.json", "dataset.json"):
        path = source_dir / filename
        if path.is_file():
            return path
    return None


def load_annotation_index(json_path: Path) -> dict[str, Any]:
    print(f"Loading JSON from {json_path} ...")
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    images_by_id = {str(image["id"]): image for image in data["images"]}
    image_id_by_name = {
        Path(image["file_name"]).name: str(image["id"])
        for image in data["images"]
    }
    bbox_by_image_id: dict[str, list[float]] = {}
    for annotation in data["annotations"]:
        bbox_by_image_id.setdefault(
            str(annotation["image_id"]), annotation["bbox"]
        )

    return {
        "images": images_by_id,
        "image_ids": image_id_by_name,
        "bboxes": bbox_by_image_id,
        "matches": {
            str(image_id): str(match_id)
            for image_id, match_id in data["id_map"].items()
        },
    }


def get_annotation(
    index: dict[str, Any], filename: str
) -> tuple[list[float], str, dict[str, Any]]:
    image_id = index["image_ids"].get(filename)
    if image_id is None:
        raise KeyError(f"Image '{filename}' not found in JSON images")

    bbox = index["bboxes"].get(image_id)
    if bbox is None:
        raise KeyError(f"No annotation found for image_id {image_id}")

    match_id = index["matches"].get(image_id)
    matched_image = index["images"].get(match_id)
    if match_id is None or matched_image is None:
        raise KeyError(f"No matched image found for image_id {image_id}")

    return bbox, image_id, matched_image


def remove_background(
    image_bgr: np.ndarray,
    bbox: list[float],
    padding: int,
    session: Any,
) -> np.ndarray:
    padded_bgr = cv2.copyMakeBorder(
        image_bgr,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    padded_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

    x, y, width, height = bbox
    prompt = {
        "sam_prompt": [
            {
                "type": "rectangle",
                "data": [
                    round(x + padding),
                    round(y + padding),
                    round(x + width + padding),
                    round(y + height + padding),
                ],
            }
        ]
    }
    carved = np.asarray(rembg.remove(padded_rgb, session=session, **prompt))
    if carved.ndim != 3 or carved.shape[2] != 4:
        raise ValueError(f"Unexpected rembg output shape: {carved.shape}")

    height, width = image_bgr.shape[:2]
    return carved[padding : padding + height, padding : padding + width]


def recenter_rgba(
    rgba: np.ndarray,
    mask: np.ndarray,
    size: int,
    border_ratio: float,
) -> np.ndarray:
    y_coords, x_coords = np.nonzero(mask)
    if y_coords.size == 0:
        return np.zeros((size, size, 4), dtype=np.uint8)

    # +1 ensures the last foreground row and column are included.
    y1, y2 = y_coords.min(), y_coords.max() + 1
    x1, x2 = x_coords.min(), x_coords.max() + 1
    cropped = rgba[y1:y2, x1:x2]
    crop_height, crop_width = cropped.shape[:2]

    target_size = max(1, round(size * (1 - border_ratio)))
    scale = target_size / max(crop_height, crop_width)
    new_width = max(1, round(crop_width * scale))
    new_height = max(1, round(crop_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def load_background(
    source_dir: Path,
    background_path: Path,
    matched_filename: str,
    fallback: np.ndarray,
) -> np.ndarray:
    if background_path.exists():
        print(f"  using existing background: {background_path}")
        return read_image(background_path)

    matched_source = source_dir / matched_filename
    if matched_source.exists():
        return read_image(matched_source)

    print(f"  warning: '{matched_filename}' not found; using current image")
    return fallback.copy()


def inpaint_background(
    image: np.ndarray,
    mask: np.ndarray,
    kernel_size: int,
    iterations: int,
    radius: int,
) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask size mismatch: image={image.shape[:2]}, mask={mask.shape}"
        )

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
    inpaint_mask = (blurred > 0).astype(np.uint8) * 255
    return cv2.inpaint(image, inpaint_mask, radius, cv2.INPAINT_NS)


def process_image(
    image_path: Path,
    dataset_dir: Path,
    index: dict[str, Any],
    session: Any,
    args: argparse.Namespace,
) -> Path:
    category_id, instance_id = parse_image_name(image_path.name)
    image = read_image(image_path)
    bbox, _, matched_image = get_annotation(index, image_path.name)

    foreground_rgba = remove_background(image, bbox, args.padding, session)
    foreground_mask = foreground_rgba[..., 3] > 0

    matched_filename = matched_image["file_name"]
    background_path = (
        dataset_dir / "background" / f"{Path(matched_filename).stem}.jpg"
    )
    background = load_background(
        dataset_dir / "source",
        background_path,
        matched_filename,
        image,
    )
    restored = inpaint_background(
        background,
        foreground_mask,
        args.dilate_kernel,
        args.dilate_iter,
        args.inpaint_radius,
    )
    write_image(background_path, restored)

    if args.recenter:
        foreground_rgba = recenter_rgba(
            foreground_rgba,
            foreground_mask,
            args.size,
            args.border_ratio,
        )

    foreground_path = (
        dataset_dir
        / "foreground"
        / f"{category_id}_{instance_id}.png"
    )
    foreground_bgra = cv2.cvtColor(foreground_rgba, cv2.COLOR_RGBA2BGRA)
    write_image(foreground_path, foreground_bgra)

    print(f"  foreground -> {foreground_path}")
    print(f"  background -> {background_path}")
    return foreground_path


def collect_images(source_dir: Path, input_value: str) -> list[Path]:
    if input_value.lower() == "all":
        images = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and IMAGE_NAME_RE.fullmatch(path.name)
        )
        if not images:
            raise FileNotFoundError(f"No valid A_B.ext images found in {source_dir}")
        return images

    if "/" in input_value or "\\" in input_value:
        raise ValueError("--img_name must be a filename, not a path")

    parse_image_name(input_value)
    image_path = source_dir / input_value
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return [image_path]


def clear_output_dirs(dataset_dir: Path) -> None:
    for name in ("foreground", "background"):
        directory = dataset_dir / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        print(f"Cleared {directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Background removal, inpainting, and recentering."
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--img_name", required=True, help="Image filename (e.g., 3_1.jpg) or 'all'")
    parser.add_argument("--json_path", help="Path to JSON annotation file (optional; auto‑searched in source/ if omitted)")
    parser.add_argument("--model", default="sam")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--border_ratio", type=float, default=0.2)
    parser.add_argument(
        "--recenter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--inpaint_radius", type=int, default=3)
    parser.add_argument("--dilate_kernel", type=int, default=5)
    parser.add_argument("--dilate_iter", type=int, default=2)
    parser.add_argument("--clear_output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_dir = dataset_dir / "source"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if args.clear_output and args.img_name.lower() != "all":
        raise ValueError("--clear_output can only be used with --img_name all")
    if not 0 <= args.border_ratio < 1:
        raise ValueError("--border_ratio must satisfy 0 <= value < 1")

    if args.clear_output:
        clear_output_dirs(dataset_dir)
    else:
        (dataset_dir / "foreground").mkdir(exist_ok=True)
        (dataset_dir / "background").mkdir(exist_ok=True)

    # JSON file is now expected inside source/ by default
    json_path = (
        Path(args.json_path).expanduser().resolve()
        if args.json_path
        else find_json(source_dir)
    )
    if json_path is None or not json_path.is_file():
        raise FileNotFoundError("Annotation JSON not found in source/; use --json_path")

    index = load_annotation_index(json_path)
    image_paths = collect_images(source_dir, args.img_name.strip())
    print(f"Found {len(image_paths)} image(s)")

    session = rembg.new_session(
        model_name=args.model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    for image_path in image_paths:
        print(f"Processing {image_path} ...")
        process_image(image_path, dataset_dir, index, session, args)


if __name__ == "__main__":
    main()