# MolProtX 主实验

运行前请按下文下载六个预训练模型，放入 `pretrained/`。

## 项目结构

```text
MolProtX/
├── configs/full.yaml             # 主实验配置，支持全部 8 个数据集
├── dataset/                     # 8 个原始数据文件
├── pretrained/                  # 模型放置目录；交付时仅保留占位文件
├── src/molprotx/                 # 模型、数据加载、训练和评价
│   └── features/                # 离线特征提取
├── scripts/
│   ├── prepare_datasets.py      # 生成或校验固定数据划分
│   ├── check_environment.py     # 检查依赖、GPU、模型、数据与缓存
│   ├── run_extract_hf.sh        # 提取指定数据集的全部 6 路特征
│   ├── run_train.sh             # 20 种子训练，自动续训和汇总
│   ├── summarize_seed_runs.py   # 汇总主实验多种子指标
│   ├── slurm_extract_hf.sbatch  # Slurm 特征提取作业
│   └── slurm_train.sbatch       # Slurm 训练作业
├── extract_hf_features.py       # 特征提取入口
├── train.py                     # 训练入口
├── pyproject.toml
└── requirements.txt
```

数据集名称：`bindingdb`、`biosnap`、`celegans`、`davis`、`drugbank`、`human`、`kiba`、`kinase`。数据划分固定为 seed=42、Train/Val/Test=7:1:2；训练的随机种子不改变数据划分。

## 安装环境

建议在 Linux + NVIDIA GPU 上运行，使用 Python 3.10 或 3.11。

```bash
cd /path/to/MolProtX
conda create -n molprotx python=3.10 -y
conda activate molprotx

python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

项目固定 `transformers==4.47.1`，不要单独升级。默认训练使用 BF16；GPU 不支持 BF16 时，将配置中的 `training.mixed_precision` 改为 `fp16`。特征提取使用大型蛋白模型，显存不足时减小配置中的 `feature_extraction.protein_batch_size`。

## 下载预训练模型

安装环境后，在联网机器上自行下载以下六个模型。放置路径相对于项目根目录，必须与 `configs/full.yaml` 一致。

| 模型及下载来源 | 放置目录 | 配置使用的权重文件 |
|---|---|---|
| [ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | `pretrained/esm2_t33_650M_UR50D/` | `model.safetensors` |
| [ProtT5-XL](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | `pretrained/prot_t5_xl_uniref50/` | `pytorch_model.bin` |
| [Ankh-base](https://huggingface.co/ElnaggarLab/ankh-base) | `pretrained/ankh-base/` | `pytorch_model.bin` |
| [MoLFormer-XL](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) | `pretrained/MoLFormer-XL-both-10pct/` | `model.safetensors` |
| [ChemBERTa-100M-MLM](https://huggingface.co/DeepChem/ChemBERTa-100M-MLM) | `pretrained/ChemBERTa-100M-MLM/` | `model.safetensors` |
| [MolDeBERTa-base-123M-mtr](https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr) | `pretrained/MolDeBERTa-base-123M-mtr/` | `model.safetensors` |

每个目录还需要同一模型版本的 `config.json`、分词器文件及配套代码，不能只放权重。MoLFormer 必须下载 **`compat-v4` 分支**，这是模型官方提供的 [Transformers v4 兼容版本](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct#example-code)，其中包括本地加载所需的自定义 Python 代码。

可以直接在项目根目录执行下面的下载命令，自动读取配置中的来源、目录和版本，仅下载所需权重及配套文件。该命令需要联网；安装项目时已通过 Transformers 安装 `huggingface_hub`。下载接口说明见 [Hugging Face 文档](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/download)。

```bash
python - <<'PY'
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

config = yaml.safe_load(Path("configs/full.yaml").read_text(encoding="utf-8"))
for spec in config["feature_extraction"]["models"].values():
    weight = "model.safetensors" if spec["use_safetensors"] else "pytorch_model.bin"
    snapshot_download(
        repo_id=spec["source_id"],
        revision=spec.get("source_revision", "main"),
        local_dir=spec["model_path"],
        allow_patterns=[weight, "*.json", "*.txt", "*.model", "*.py", "README.md"],
    )
PY
```

也可以在其他联网机器下载，再把这六个完整目录复制到服务器的 `pretrained/` 下。特征提取和训练始终离线读取本地文件，不会自动联网补齐模型。

## 开始训练

以 Human 数据集为例，其他数据集替换 `human` 即可。

```bash
# 1. 从原始数据生成固定划分（首次运行）
python scripts/prepare_datasets.py --dataset all

# 2. 检查环境并从本地模型离线提取全部6路特征
bash scripts/run_extract_hf.sh --dataset human

# 3. 检查缓存完整性
python scripts/check_environment.py --config configs/full.yaml --dataset human --require-features

# 4. 启动单种子主实验
python train.py --config configs/full.yaml --dataset human --seed 42
```

首次运行必须先下载六个模型、生成划分，再提取特征。已有划分可用 `python scripts/prepare_datasets.py --dataset all --check` 校验。模型从本目录的 `pretrained/` 离线读取；训练使用缓存，不重复执行六个预训练模型。再次提取时会校验并跳过已有的有效缓存。

完整 20 种子主实验（替代上面的单种子训练命令）：

```bash
bash scripts/run_train.sh --dataset human
```

默认种子：`0 1 7 11 13 17 19 23 29 31 37 42 43 47 53 59 67 71 73 97`。脚本依次运行，跳过已完成的种子、从已有断点续训，并汇总指标。只跑指定种子也可使用：

```bash
MOLPROTX_SEEDS="42" bash scripts/run_train.sh --dataset human
```

单独恢复中断的训练：

```bash
python train.py --dataset human --seed 42 --resume outputs/human/MolProtX/seed42/checkpoints/last_checkpoint.pt
```

特征提取和训练均支持 `--device cuda:1` 指定 GPU。默认最多 1000 个 epoch，监控验证集 AUPRC，连续 20 个 epoch 无足够改善时早停；具体参数见 `configs/full.yaml`。

## 运行后生成的目录

- `dataset/splits/<dataset>/`：固定划分及其校验清单。
- `artifacts/features/<dataset>/`：离线特征缓存。
- `outputs/<dataset>/MolProtX/seed<seed>/`：权重、日志、指标和预测。
- `outputs/<dataset>/MolProtX/`：多种子汇总文件。
