# Boosting Cross-Domain Few-Shot Object Detection with Generative 3D Geometry Priors

Official implementation of **“Boosting Cross-Domain Few-Shot Object Detection with Generative 3D Geometry Priors.”**

This repository contains:

* **DreamGaussian** for multi-view object generation and data augmentation.
* **CDFSOD-benchmark** for cross-domain few-shot object detection.

## Project Structure

```text
cdfsod/
├── dreamgaussian/
├── cdfsod-main/
├── generate/
├── setup.sh
└── README.md
```

`setup.sh` copies the custom scripts in `generate/` into the root directory of `dreamgaussian/`.

## Installation

### Clone the Repository

```bash
git clone --recurse-submodules https://github.com/xSheep123/cdfsod.git
cd cdfsod
bash setup.sh
```

If the submodules were not cloned:

```bash
git submodule update --init --recursive
bash setup.sh
```

### DreamGaussian Environment

Follow the official installation guide:

* [DreamGaussian Installation](https://github.com/dreamgaussian/dreamgaussian#install)

```bash
cd dreamgaussian
```

### CDFSOD Environment

```bash
cd ..
conda create -n cdfsod python=3.9 -y
conda activate cdfsod
pip install -r cdfsod-main/requirements.txt
pip install -e ./cdfsod-main
```

We recommend using separate environments for DreamGaussian and CDFSOD.

## Data Preparation

Download and prepare the datasets following:

* [CDFSOD-benchmark Datasets](https://github.com/lovelyqian/CDFSOD-benchmark/tree/main#datasets)

This project uses two directories:

* `dataset_dir`: DreamGaussian workspace and intermediate results.
* `output_dir`: original and augmented detection dataset.

Reorganize the dataset:

```bash
python rewrite.py \
  --dataset_dir /path/to/dataset_dir \
  --output_dir /path/to/output_dir
```

`rewrite.py` reorganizes `output_dir` for detector training and copies the required images and annotations to `dataset_dir/source/`.

### DreamGaussian Workspace

```text
dataset_dir/
├── source/
├── foreground/
├── background/
├── logs/
│   └── A_B/
│       └── result.mp4
└── frame/
    └── A_B/
        └── frame_0.jpg
```

### Detection Dataset

```text
output_dir/
├── train_{shot}shot/
│   ├── original images
│   └── generate_{...}.jpg
└── annotations/
    └── {shot}_shot.json
```

## Data Generation

Activate the DreamGaussian environment:

```bash
conda activate dreamgaussian
cd dreamgaussian
```

The following commands use ArTaxOr 1-shot as an example.

### 1. Extract Foregrounds and Generate Backgrounds

```bash
python prepocess.py \
  --dataset_dir /path/to/dataset_dir \
  --img_name all \
  --json_path /path/to/dataset_dir/source/1-shot_new.json
```

### 2. Generate Multi-View Videos

```bash
bash result
```

### 3. Select Candidate Frames

```bash
python rank_frames.py \
  --dataset_dir /path/to/dataset_dir \
  --img_name all \
  --dataset ArTaxOr \
  --json_path /path/to/dataset_dir/source/1-shot_new.json \
  --text_dir ./text
```

### 4. Compose Augmented Images

```bash
python compose.py \
  --dataset_dir /path/to/dataset_dir \
  --output_dir /path/to/output_dir \
  --dataset ArTaxOr \
  --shot 1 \
  --num_categories 7
```

For other datasets, update `--dataset`, `--shot`, `--num_categories`, and `--json_path`.

## Detector Training

Activate the CDFSOD environment:

```bash
cd ../cdfsod-main
conda activate cdfsod
```

### 1. Preparation

Modify the dataset paths in `detectron2/data/datasets/builtin.py` and download the required weights following the [CDFSOD Running Instructions](https://github.com/lovelyqian/CDFSOD-benchmark#run-cd-vito).

### 2. Generate Class Prototypes

```bash
bash build_prototypes.sh
```

### 3. Train and Evaluate

```bash
bash main_results.sh
```


## Notes

* Use absolute paths for `dataset_dir` and `output_dir`.
* Rerun `bash setup.sh` after updating scripts in `generate/`.
* Make sure the image and directory names follow the expected format.
* Verify that `prepocess.py` and `result` match the actual filenames in the repository.

## Acknowledgements

This project is based on:

* [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark)
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)

We thank the authors for releasing their code.
