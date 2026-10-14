#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法：bash scripts/run_train.sh --dataset {bindingdb|biosnap|celegans|davis|drugbank|human|kiba|kinase}"
}

if [[ $# -eq 1 && "${1}" != --* ]]; then
  DATASET="${1}"
elif [[ $# -eq 2 && "${1}" == "--dataset" ]]; then
  DATASET="${2}"
else
  usage >&2
  exit 2
fi
DATASET="${DATASET,,}"
case "${DATASET}" in
  bindingdb|biosnap|celegans|davis|drugbank|human|kiba|kinase) ;;
  *) usage >&2; exit 2 ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONHASHSEED="${MOLPROTX_SEED:-42}"

python scripts/prepare_datasets.py --dataset "${DATASET}" --check
python scripts/check_environment.py \
  --config configs/full.yaml \
  --dataset "${DATASET}" \
  --require-features

DEFAULT_SEEDS=(0 1 7 11 13 17 19 23 29 31 37 42 43 47 53 59 67 71 73 97)
if [[ -n "${MOLPROTX_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${MOLPROTX_SEEDS}"
  SUMMARY_STEM="${MOLPROTX_SUMMARY_STEM:-summary_custom_seeds}"
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
  SUMMARY_STEM="summary_20seeds"
fi

echo "数据集=${DATASET}；将依次运行 ${#SEEDS[@]} 个随机种子：${SEEDS[*]}"
for INDEX in "${!SEEDS[@]}"; do
  SEED="${SEEDS[${INDEX}]}"
  export PYTHONHASHSEED="${SEED}"
  OUTPUT_DIR="${PROJECT_DIR}/outputs/${DATASET}/MolProtX/seed${SEED}"
  FINAL_METRICS="${OUTPUT_DIR}/metrics_final.json"
  LAST_CHECKPOINT="${OUTPUT_DIR}/checkpoints/last_checkpoint.pt"
  echo "[$((INDEX + 1))/${#SEEDS[@]}] seed=${SEED} | output=${OUTPUT_DIR}"

  if [[ -f "${FINAL_METRICS}" ]]; then
    echo "seed=${SEED} 已完成，跳过训练并纳入汇总。"
  elif [[ -f "${LAST_CHECKPOINT}" ]]; then
    echo "seed=${SEED} 检测到断点，自动续训。"
    python train.py \
      --config configs/full.yaml \
      --dataset "${DATASET}" \
      --seed "${SEED}" \
      --resume "${LAST_CHECKPOINT}"
  else
    python train.py \
      --config configs/full.yaml \
      --dataset "${DATASET}" \
      --seed "${SEED}"
  fi

  python scripts/summarize_seed_runs.py \
    --config configs/full.yaml \
    --dataset "${DATASET}" \
    --seeds "${SEEDS[@]}" \
    --output-stem "${SUMMARY_STEM}" \
    --allow-incomplete
done

python scripts/summarize_seed_runs.py \
  --config configs/full.yaml \
  --dataset "${DATASET}" \
  --seeds "${SEEDS[@]}" \
  --output-stem "${SUMMARY_STEM}"
echo "全部随机种子实验完成：outputs/${DATASET}/MolProtX/${SUMMARY_STEM}.md"
