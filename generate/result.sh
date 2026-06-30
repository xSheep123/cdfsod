set -euo pipefail

GPU=4
DATASET_ROOT="/path/to/dataset_dir"

datasets=(
  clipart1k
  ArTaxOr
  FISH
  UODD
)

shots=(
  1
  5
  10
)

declare -A categories=(
  [clipart1k]="1:20"
  [ArTaxOr]="1:7"
  [FISH]="1:1"
  [UODD]="2:2"
)

for dataset in "${datasets[@]}"; do
  IFS=: read -r start end <<< "${categories[$dataset]}"
  for shot in "${shots[@]}"; do
    for ((category=start; category<=end; category++)); do
      for ((index=1; index<=shot; index++)); do
        name="${category}_${index}"
        input="dataset/${num_shot}-shot/${dataset}/foreground/${category}_${shot}.png"
        save="${shot}-shot/${dataset}/${name}/${name}"
        output="${DATASET_ROOT}/${shot}-shot/${dataset}/logs/${name}/result.mp4"

        [[ -f "$input" ]] || continue
        mkdir -p "$(dirname "$output")"

        CUDA_VISIBLE_DEVICES=$GPU python main.py --config configs/image.yaml input="$input" save_path="$save" force_cuda_rast=True
        CUDA_VISIBLE_DEVICES=$GPU python main2.py --config configs/image.yaml input="$input" save_path="$save" force_cuda_rast=True
        CUDA_VISIBLE_DEVICES=$GPU kire "logs/${save}.obj" --save_video "$output" --wogui --force_cuda_rast
      done
    done
  done
done