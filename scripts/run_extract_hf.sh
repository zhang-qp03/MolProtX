#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法：bash scripts/run_extract_hf.sh --dataset {bindingdb|biosnap|celegans|davis|drugbank|human|kiba|kinase}"
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/prepare_datasets.py --dataset "${DATASET}" --check
python scripts/check_environment.py \
  --config configs/full.yaml \
  --dataset "${DATASET}"
python extract_hf_features.py \
  --config configs/full.yaml \
  --dataset "${DATASET}" \
  --models esm2 prott5 ankh molformer chemberta moldeberta
