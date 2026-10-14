#!/usr/bin/env bash
#SBATCH --job-name=molprotx-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=molprotx-embed-%j.out
#SBATCH --error=molprotx-embed-%j.err

set -euo pipefail
: "${MOLPROTX_ENV:?请设置 MOLPROTX_ENV 为 Conda 环境名称}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MOLPROTX_ENV}"
cd "${SLURM_SUBMIT_DIR:?请从 MolProtX 项目根目录提交作业}"
DATASET="${1:-${MOLPROTX_DATASET:-}}"
: "${DATASET:?请通过 sbatch scripts/slurm_extract_hf.sbatch <dataset> 或 MOLPROTX_DATASET 指定数据集}"
bash scripts/run_extract_hf.sh --dataset "${DATASET}"
