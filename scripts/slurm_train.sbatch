#!/usr/bin/env bash
#SBATCH --job-name=molprotx-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=molprotx-train-%j.out
#SBATCH --error=molprotx-train-%j.err

set -euo pipefail
: "${MOLPROTX_ENV:?请设置 MOLPROTX_ENV 为 Conda 环境名称}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${MOLPROTX_ENV}"
cd "${SLURM_SUBMIT_DIR:?请从 MolProtX 项目根目录提交作业}"
DATASET="${1:-${MOLPROTX_DATASET:-}}"
: "${DATASET:?请通过 sbatch scripts/slurm_train.sbatch <dataset> 或 MOLPROTX_DATASET 指定数据集}"
bash scripts/run_train.sh --dataset "${DATASET}"
