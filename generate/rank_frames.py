#!/usr/bin/env python3
"""Select Top-K frames from result.mp4 using CLIP-based scores.

Dataset layout:
    dataset_dir/
    ├── source/
    │   ├── A_B.ext
    │   └── *shot*.json
    ├── background/
    │   └── MATCHED_IMAGE.jpg
    ├── logs/
    │   └── A_B/
    │       └── result.mp4
    └── frame/
        └── A_B/
            ├── frame_0.jpg
            └── scores.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_PATTERN = re.compile(r"^\d+_\d+$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def find_json(source_dir: Path) -> Path:
    shot_files = sorted(
        p for p in source_dir.glob("*.json")
        if "shot" in p.name.lower()
    )
    if shot_files:
        return shot_files[0]

    for name in ("annotations.json", "dataset.json"):
        p = source_dir / name
        if p.is_file():
            return p

    raise FileNotFoundError(f"No annotation JSON found in {source_dir}")


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {str(img["id"]): img for img in data["images"]}
    images_by_name = {
        Path(img["file_name"]).name: img
        for img in data["images"]
    }

    annos_by_img = {}
    for ann in data["annotations"]:
        annos_by_img.setdefault(str(ann["image_id"]), []).append(ann)

    id_map = {str(k): str(v) for k, v in data["id_map"].items()}
    return images_by_id, images_by_name, annos_by_img, id_map


def load_descriptions(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Description file not found: {path}")

    desc = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid description: {line}")
        cid, d = line.split(":", 1)
        desc[cid.strip()] = d.strip()
    return desc


def collect_source_images(source_dir: Path, image_name: str) -> list[Path]:
    if image_name.lower() != "all":
        if "/" in image_name or "\\" in image_name:
            raise ValueError("--img_name must be a filename, not a path")

        p = source_dir / image_name
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported format: {p.suffix}")
        if not IMAGE_PATTERN.fullmatch(p.stem):
            raise ValueError(f"Image must be A_B.ext: {p.name}")
        return [p]

    images = [
        p for p in source_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and IMAGE_PATTERN.fullmatch(p.stem)
    ]
    images.sort(key=lambda p: tuple(map(int, p.stem.split("_"))))
    if not images:
        raise FileNotFoundError(f"No A_B.ext images in {source_dir}")
    return images


# ── Image processing helpers ─────────────────────────────────────────────────

def crop_white_border(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray < threshold)
    if not len(xs):
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste_candidate(background: np.ndarray, candidate: np.ndarray, bbox) -> np.ndarray:
    """Paste a white-background candidate frame into an XYWH box."""
    result = background.copy()
    candidate = crop_white_border(candidate)

    x, y, w, h = map(round, bbox)
    if candidate.size == 0 or w <= 0 or h <= 0:
        return result

    scale = min(w / candidate.shape[1], h / candidate.shape[0])
    nw = max(1, round(candidate.shape[1] * scale))
    nh = max(1, round(candidate.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    candidate = cv2.resize(candidate, (nw, nh), interpolation=interp)

    mask = (cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY) < 245)
    left = round(x + w / 2 - nw / 2)
    top = round(y + h / 2 - nh / 2)
    right = left + nw
    bottom = top + nh

    dst_l, dst_t = max(0, left), max(0, top)
    dst_r, dst_b = min(result.shape[1], right), min(result.shape[0], bottom)
    if dst_l >= dst_r or dst_t >= dst_b:
        return result

    src_l, src_t = dst_l - left, dst_t - top
    src_r, src_b = src_l + dst_r - dst_l, src_t + dst_b - dst_t

    region = result[dst_t:dst_b, dst_l:dst_r]
    source = candidate[src_t:src_b, src_l:src_r]
    source_mask = mask[src_t:src_b, src_l:src_r]
    region[source_mask] = source[source_mask]
    return result


def to_clip_tensor(image: np.ndarray, preprocess) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return preprocess(Image.fromarray(rgb))


def read_selected_frames(video_path: Path, frame_ids: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for fid in frame_ids:
        capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"Cannot read frame {fid} from {video_path}")
        frames.append(frame)
    capture.release()
    return frames


# ── CLIP Retriever ───────────────────────────────────────────────────────────

class CLIPFrameRetriever:
    def __init__(self, model_name: str, device: str, weights: tuple[float, float, float]):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.sds_weight, self.image_weight, self.text_weight = weights

    @torch.inference_mode()
    def encode_query(self, text: str, source_image: np.ndarray):
        text_tokens = clip.tokenize([text]).to(self.device)
        text_feat = self.model.encode_text(text_tokens).float()

        img_tensor = to_clip_tensor(source_image, self.preprocess).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_tensor).float()
        return text_feat, img_feat

    @torch.inference_mode()
    def score_video(self, video_path: Path, text_feat: torch.Tensor, img_feat: torch.Tensor,
                    background: np.ndarray, bbox, batch_size: int, frame_stride: int):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_ids = []
        raw_features = []
        img_scores = []
        text_scores = []
        raw_batch = []
        comp_batch = []
        batch_ids = []
        frame_id = 0

        def flush():
            if not raw_batch:
                return
            raw_inp = torch.stack(raw_batch).to(self.device)
            comp_inp = torch.stack(comp_batch).to(self.device)
            raw_feat = self.model.encode_image(raw_inp).float()
            comp_feat = self.model.encode_image(comp_inp).float()

            raw_features.append(raw_feat.cpu())
            text_scores.append(F.cosine_similarity(raw_feat, text_feat.expand_as(raw_feat), dim=1).cpu())
            img_scores.append(F.cosine_similarity(comp_feat, img_feat.expand_as(comp_feat), dim=1).cpu())
            frame_ids.extend(batch_ids)

            raw_batch.clear()
            comp_batch.clear()
            batch_ids.clear()

        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_id % frame_stride == 0:
                raw_batch.append(to_clip_tensor(frame, self.preprocess))
                composite = paste_candidate(background, frame, bbox)
                comp_batch.append(to_clip_tensor(composite, self.preprocess))
                batch_ids.append(frame_id)

                if len(raw_batch) == batch_size:
                    flush()
            frame_id += 1

        flush()
        capture.release()

        if not raw_features:
            raise RuntimeError(f"Video contains no valid frames: {video_path}")

        raw_features = torch.cat(raw_features)
        mean_feat = raw_features.mean(dim=0, keepdim=True)
        sds_scores = F.cosine_similarity(raw_features, mean_feat, dim=1)
        total = (self.sds_weight * sds_scores
                 + self.image_weight * torch.cat(img_scores)
                 + self.text_weight * torch.cat(text_scores))
        return torch.tensor(frame_ids), total

    def retrieve(self, text: str, source_image: np.ndarray, background: np.ndarray,
                 bbox, video_path: Path, topk: int, batch_size: int, frame_stride: int):
        text_feat, img_feat = self.encode_query(text, source_image)
        frame_ids, scores = self.score_video(
            video_path, text_feat, img_feat, background, bbox, batch_size, frame_stride
        )
        top_scores, pos = torch.topk(scores, k=min(topk, len(scores)), largest=True, sorted=True)
        frames = read_selected_frames(video_path, frame_ids[pos].tolist())
        return frame_ids[pos], top_scores, frames


# ── Annotation resolution ────────────────────────────────────────────────────

def resolve_sample(source_path: Path, images_by_id, images_by_name, annos_by_img, id_map):
    img_info = images_by_name.get(source_path.name)
    if img_info is None:
        raise KeyError(f"{source_path.name} not found in JSON images")

    img_id = str(img_info["id"])
    annos = annos_by_img.get(img_id, [])
    if not annos:
        raise KeyError(f"No annotation for {source_path.name}")

    annotation = annos[0]
    match_id = id_map.get(img_id)
    matched = images_by_id.get(match_id) if match_id else None
    if matched is None:
        raise KeyError(f"No matched image for {source_path.name}")
    return annotation, img_info, matched


# ── Main per-image logic ─────────────────────────────────────────────────────

def process_image(source_path: Path, args: argparse.Namespace,
                  retriever: CLIPFrameRetriever, dataset,
                  descriptions: dict[str, str]) -> None:
    images_by_id, images_by_name, annos_by_img, id_map = dataset
    annotation, img_info, matched = resolve_sample(
        source_path, images_by_id, images_by_name, annos_by_img, id_map
    )

    file_stem = Path(img_info["file_name"]).stem
    matched_stem = Path(matched["file_name"]).stem

    desc = descriptions.get(str(annotation["category_id"]))
    if not desc:
        raise KeyError(f"No description for category {annotation['category_id']}")

    bg_path = args.dataset_dir / "background" / f"{matched_stem}.jpg"
    video_path = args.dataset_dir / "logs" / file_stem / "result.mp4"

    source_img = read_image(source_path)
    background = read_image(bg_path)

    frame_ids, scores, frames = retriever.retrieve(
        text=desc,
        source_image=source_img,
        background=background,
        bbox=annotation["bbox"],
        video_path=video_path,
        topk=args.topk,
        batch_size=args.batch_size,
        frame_stride=args.frame_stride,
    )

    save_dir = args.output_dir / file_stem
    save_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for rank, (fid, score, frame) in enumerate(zip(frame_ids, scores, frames)):
        out = save_dir / f"frame_{rank}.jpg"
        if not cv2.imwrite(str(out), frame):
            raise RuntimeError(f"Cannot save frame: {out}")
        records.append({"rank": rank, "frame_id": int(fid), "score": float(score)})
        print(f"  frame={int(fid):4d}, score={float(score):.4f} -> {out}")

    (save_dir / "scores.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select Top-K frames from logs/A_B/result.mp4 using CLIP."
    )
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--img_name", required=True, help="Source image filename or 'all'.")
    p.add_argument("--dataset", required=True, help="Dataset name (text/<dataset>.txt).")
    p.add_argument("--output_dir", default="frame", help="Output subdirectory inside dataset_dir.")
    p.add_argument("--json_path", type=Path, help="Annotation JSON path (auto-search if omitted).")
    p.add_argument("--text_dir", type=Path, default=Path("text"))
    p.add_argument("--clip_model", default="ViT-B/32")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--weights", nargs=3, type=float, default=(0.1, 0.4, 0.8),
                   metavar=("SDS", "IMAGE", "TEXT"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=666)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.dataset_dir / args.output_dir

    if min(args.topk, args.batch_size, args.frame_stride) <= 0:
        raise ValueError("topk, batch_size, and frame_stride must be positive")

    source_dir = args.dataset_dir / "source"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    json_path = args.json_path.expanduser().resolve() if args.json_path else find_json(source_dir)
    text_path = args.text_dir.expanduser().resolve() / f"{args.dataset}.txt"

    dataset = load_dataset(json_path)
    descriptions = load_descriptions(text_path)
    source_images = collect_source_images(source_dir, args.img_name)

    retriever = CLIPFrameRetriever(
        model_name=args.clip_model,
        device=args.device,
        weights=tuple(args.weights),
    )

    print(f"JSON: {json_path}")
    print(f"Found {len(source_images)} source image(s).")

    for idx, src_path in enumerate(source_images, 1):
        print(f"\n[{idx}/{len(source_images)}] Processing {src_path.name}")
        try:
            process_image(src_path, args, retriever, dataset, descriptions)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

