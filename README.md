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
├── prepare_data.py
├── setup.sh
└── README.md
```

`setup.sh` copies the custom scripts in `generate/` into the root directory of `dreamgaussian/`.

## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/xSheep123/cdfsod.git
cd cdfsod
bash setup.sh
```

If the repository was cloned without submodules, run:

```bash
git submodule update --init --recursive
bash setup.sh
```

### 2. DreamGaussian Environment

Follow the official installation guide: [DreamGaussian Installation](https://github.com/dreamgaussian/dreamgaussian#install)

```bash
cd dreamgaussian
```

### 3. CDFSOD Environment

```bash
cd ..
conda create -n cdfsod python=3.9 -y
conda activate cdfsod
pip install -r cdfsod-main/requirements.txt
pip install -e ./cdfsod-main
```

Separate environments are recommended for DreamGaussian and CDFSOD.

## Data Preparation

Download the datasets following: [CDFSOD-benchmark Datasets](https://github.com/lovelyqian/CDFSOD-benchmark/tree/main#datasets)

Prepare the downloaded datasets using:

```bash
python prepare_data.py \
  --data_root /path/to/data_root \
  --work_dir /path/to/work_dir
```

Arguments:

* `data_root`: root directory of the downloaded detection datasets and the final augmented datasets.
* `work_dir`: DreamGaussian workspace for source data and intermediate results.

To process specific datasets or shot settings:

```bash
python prepare_data.py \
  --data_root /path/to/data_root \
  --work_dir /path/to/work_dir \
  --datasets ArTaxOr UODD \
  --shots 1 5
```

For a dataset and shot setting, the corresponding DreamGaussian workspace is:

```text
work_dir/{shot}-shot/{dataset}/
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

The detection dataset is organized as:

```text
data_root/{dataset}/
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

The following commands use **ArTaxOr 1-shot** as an example:

```text
dataset_dir = /path/to/work_dir/1-shot/ArTaxOr
output_dir  = /path/to/data_root/ArTaxOr
```

### 1. Extract Foregrounds and Generate Backgrounds

```bash
python prepocess.py \
  --dataset_dir /path/to/work_dir/1-shot/ArTaxOr \
  --img_name all \
  --json_path /path/to/work_dir/1-shot/ArTaxOr/source/1-shot_new.json
```

### 2. Generate Multi-View Videos

```bash
bash result
```

### 3. Select Candidate Frames

```bash
python rank_frames.py \
  --dataset_dir /path/to/work_dir/1-shot/ArTaxOr \
  --img_name all \
  --dataset ArTaxOr \
  --json_path /path/to/work_dir/1-shot/ArTaxOr/source/1-shot_new.json \
  --text_dir ./text
```

### 4. Compose Augmented Images

```bash
python compose.py \
  --dataset_dir /path/to/work_dir/1-shot/ArTaxOr \
  --output_dir /path/to/data_root/ArTaxOr \
  --dataset ArTaxOr \
  --shot 1 \
  --num_categories 7
```

For other datasets, update `--dataset`, `--shot`, `--num_categories`, and the corresponding paths.

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

* Use absolute paths for `data_root` and `work_dir`.
* Rerun `bash setup.sh` after updating scripts in `generate/`.
* Make sure image and directory names follow the expected format.
* Verify that `prepocess.py` and `result` match the actual filenames in the repository.

## Acknowledgements

This project is based on:

* [CDFSOD-benchmark](https://github.com/lovelyqian/CDFSOD-benchmark)
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)

We thank the authors for releasing their code.
