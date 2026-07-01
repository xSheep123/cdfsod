import argparse
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path

DEFAULT_DATASETS = ["ArTaxOr", "clipart1k", "UODD", "FISH"]
DEFAULT_SHOTS = [1, 5, 10]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for DreamGaussian and CDFSOD."
    )
    parser.add_argument(
        "--data_root",type=Path,required=True,help="Root directory of the detection datasets.",
    )
    parser.add_argument(
        "--work_dir",type=Path,required=True,help="Workspace for DreamGaussian data generation.",
    )
    parser.add_argument("--datasets",nargs="+",default=DEFAULT_DATASETS,help="Datasets to process.",
    )
    parser.add_argument("--shots",nargs="+",type=int,default=DEFAULT_SHOTS,help="Few-shot settings to process.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def find_image(train_dir, file_name):
    image_path = train_dir / file_name
    if image_path.exists():
        return image_path

    matches = list(train_dir.rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(f"Image not found: {file_name}")

    raise RuntimeError(f"Multiple images found with the same name: {file_name}")


def reindex_images(data):
    """Keep annotated images and reindex their IDs to 1, 2, ..., N."""
    image_dict = {image["id"]: image for image in data["images"]}
    used_ids = {annotation["image_id"] for annotation in data["annotations"]}

    missing_ids = used_ids - set(image_dict)
    if missing_ids:
        raise ValueError(f"Annotations reference missing image IDs: {missing_ids}")

    ordered_ids = [
        image["id"]
        for image in data["images"]
        if image["id"] in used_ids
    ]

    id_map = {
        old_id: new_id
        for new_id, old_id in enumerate(ordered_ids, start=1)
    }

    new_data = copy.deepcopy(data)
    new_data["images"] = []

    for old_id in ordered_ids:
        image = copy.deepcopy(image_dict[old_id])
        image["id"] = id_map[old_id]
        new_data["images"].append(image)

    for annotation in new_data["annotations"]:
        annotation["image_id"] = id_map[annotation["image_id"]]

    new_data.pop("id_map", None)
    return new_data


def process_split(dataset, shot, data_root, work_dir):
    dataset_dir = data_root / dataset
    train_dir = dataset_dir / "train"
    shot_train_dir = dataset_dir / f"train_{shot}shot"
    source_dir = work_dir / f"{shot}-shot" / dataset / "source"
    annotation_path = dataset_dir / "annotations" / f"{shot}.json"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    data = reindex_images(load_json(annotation_path))

    shutil.rmtree(shot_train_dir, ignore_errors=True)
    shutil.rmtree(source_dir, ignore_errors=True)

    shot_train_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    image_dict = {image["id"]: image for image in data["images"]}
    image_paths = {}

    # Copy the original images for detector training.
    for image in data["images"]:
        source_path = find_image(train_dir, image["file_name"])
        target_path = shot_train_dir / image["file_name"]

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        image_paths[image["id"]] = source_path

    # Save the annotations with continuous image IDs.
    save_json(data, source_dir / f"{shot}-shot.json")

    # Preserve existing image IDs and append new images for repeated annotations.
    new_images = copy.deepcopy(data["images"])
    new_image_dict = {image["id"]: image for image in new_images}
    new_annotations = []
    new_id_map = {}

    category_count = defaultdict(int)
    image_annotation_count = defaultdict(int)
    max_image_id = max(image_dict, default=0)

    for annotation in data["annotations"]:
        old_image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        category_count[category_id] += 1
        image_annotation_count[old_image_id] += 1

        source_path = image_paths[old_image_id]
        suffix = source_path.suffix or ".jpg"
        new_name = f"{category_id}_{category_count[category_id]}{suffix}"

        shutil.copy2(source_path, source_dir / new_name)

        if image_annotation_count[old_image_id] == 1:
            # Keep the original image ID for the first annotation.
            new_image_id = old_image_id
            new_image_dict[old_image_id]["file_name"] = new_name
        else:
            # Append a new image entry for each additional annotation.
            max_image_id += 1
            new_image_id = max_image_id

            new_image = copy.deepcopy(image_dict[old_image_id])
            new_image["id"] = new_image_id
            new_image["file_name"] = new_name
            new_images.append(new_image)

        new_annotation = copy.deepcopy(annotation)
        new_annotation["image_id"] = new_image_id
        new_annotations.append(new_annotation)

        # Current image ID -> image ID in {shot}-shot.json.
        new_id_map[new_image_id] = old_image_id

    new_data = copy.deepcopy(data)
    new_data["images"] = new_images
    new_data["annotations"] = new_annotations
    new_data["id_map"] = new_id_map

    save_json(new_data, source_dir / f"{shot}-shot_new.json")

    print(
        f"Completed: {dataset} {shot}-shot | "
        f"images: {len(data['images'])} -> {len(new_images)}, "
        f"annotations: {len(new_annotations)}"
    )


def main():
    args = parse_args()

    for dataset in args.datasets:
        for shot in args.shots:
            try:
                process_split(
                    dataset=dataset,
                    shot=shot,
                    data_root=args.data_root,
                    work_dir=args.work_dir,
                )
            except Exception as error:
                print(f"Failed: {dataset} {shot}-shot | {error}")


if __name__ == "__main__":
    main()