from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer, T5Tokenizer

from molprotx.config import DATASET_NAMES, load_config, normalize_dataset_name, resolved_config
from molprotx.data import validate_fixed_split
from molprotx.feature_store import FeatureStore
from molprotx.features.common import collect_unique_entities, local_checkpoint_signature


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 MolProtX 服务器环境、数据文件和特征缓存。")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--dataset", required=True, type=normalize_dataset_name, choices=DATASET_NAMES
    )
    parser.add_argument("--require-features", action="store_true", help="缺少任意特征文件时返回失败。")
    args = parser.parse_args()
    config, project_root = load_config(args.config, dataset=args.dataset)
    config = resolved_config(config, project_root)

    print("=== 软件环境 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"GPU {index}: {props.name}, VRAM={props.total_memory / 1024**3:.1f} GiB")
    required_packages = {
        "transformers": "==4.47.1",
        "tokenizers": ">=0.21,<0.22",
        "accelerate": ">=0.34,<2",
        "safetensors": ">=0.4,<1",
        "sentencepiece": ">=0.2,<1",
        "protobuf": ">=4.25,<6",
        "numpy": ">=1.26,<3",
        "pandas": ">=2.1,<3",
        "openpyxl": ">=3.1,<4",
        "scikit-learn": ">=1.4,<2",
        "PyYAML": ">=6,<7",
        "tqdm": ">=4.66,<5",
        "tensorboard": ">=2.16,<3",
    }
    installed_versions: dict[str, str] = {}
    for package in required_packages:
        try:
            version = importlib.metadata.version(package)
            installed_versions[package] = version
            print(f"{package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: 未安装")

    errors: list[str] = []
    python_version = Version(".".join(str(value) for value in sys.version_info[:3]))
    if python_version not in SpecifierSet(">=3.10,<3.13"):
        errors.append(f"Python {python_version} 不满足 >=3.10,<3.13")
    if Version(torch.__version__.split("+", maxsplit=1)[0]) not in SpecifierSet(">=2.2,<2.6"):
        errors.append(f"PyTorch {torch.__version__} 不满足 >=2.2,<2.6")
    for package, requirement in required_packages.items():
        installed = installed_versions.get(package)
        if installed is None:
            errors.append(f"{package} 未安装")
        elif Version(installed) not in SpecifierSet(requirement):
            errors.append(f"{package} {installed} 不满足 {requirement}")
    requested_devices = {
        str(config["feature_extraction"]["device"]),
        str(config["training"]["device"]),
    }
    if any(device.startswith("cuda") for device in requested_devices) and not torch.cuda.is_available():
        errors.append("配置要求 CUDA，但 PyTorch 未检测到可用 NVIDIA GPU")
    if (
        str(config["training"]["mixed_precision"]) == "bf16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        errors.append("配置要求 BF16，但当前 GPU 不支持；请将 mixed_precision 改为 fp16")
    if errors:
        raise RuntimeError("软件环境不符合主实验要求：" + "；".join(errors))

    print("\n=== 本地预训练模型 ===")
    model_specs = config["feature_extraction"]["models"]
    for name, spec in model_specs.items():
        model_path = Path(spec["model_path"])
        if name == "molformer":
            modeling_file = model_path / "modeling_molformer.py"
            if not modeling_file.is_file():
                raise FileNotFoundError(f"MolFormer 缺少本地模型代码：{modeling_file}")
            modeling_source = modeling_file.read_text(encoding="utf-8")
            if "transformers.masking_utils" in modeling_source:
                raise ValueError(
                    "MolFormer 的 modeling_molformer.py 来自 Transformers v5/main 分支，"
                    "与本项目固定的 Transformers 4.47.1 不兼容。"
                    "请换成官方 compat-v4 分支中的同名文件。"
                )
        signature, weight_files = local_checkpoint_signature(
            model_path, bool(spec.get("use_safetensors", True))
        )
        raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        actual_dim = raw_config.get("hidden_size", raw_config.get("d_model"))
        expected_dim = int(spec["expected_dim"])
        if actual_dim is None or int(actual_dim) != expected_dim:
            raise ValueError(
                f"{name} 的 config.json 隐藏维度为 {actual_dim}，配置期望 {expected_dim}。"
            )
        max_positions = raw_config.get("max_position_embeddings")
        position_offset = 2 if raw_config.get("model_type") == "roberta" else 0
        if max_positions is not None and int(spec["max_length"]) + position_offset > int(max_positions):
            raise ValueError(
                f"{name} 的 max_length={spec['max_length']} 超过模型可用位置长度 "
                f"{int(max_positions) - position_offset}。"
            )
        common_kwargs = {
            "local_files_only": True,
            "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        }
        AutoConfig.from_pretrained(model_path, **common_kwargs)
        if name == "prott5":
            T5Tokenizer.from_pretrained(model_path, legacy=True, local_files_only=True)
        else:
            AutoTokenizer.from_pretrained(model_path, **common_kwargs)
        print(
            f"{name}: {model_path} | source={spec['source_id']} | dim={actual_dim} | "
            f"weights={weight_files} | signature={signature[:12]}..."
        )

    print("\n=== 数据文件 ===")
    data_config = config["data"]
    manifest = validate_fixed_split(data_config)
    print(
        f"fixed split: seed={manifest['split_seed']} | ratios={manifest['split_ratios']} | "
        f"source_sha256={manifest['source_sha256'][:12]}..."
    )
    required_columns = {
        data_config["smiles_column"],
        data_config["sequence_column"],
        data_config["label_column"],
    }
    required_columns.update(
        column
        for column in (data_config.get("drug_id_column"), data_config.get("protein_id_column"))
        if column is not None
    )
    for split in ("train", "val", "test"):
        path = Path(data_config[f"{split}_csv"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=list(required_columns))
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少列：{sorted(missing)}")
        labels = sorted(frame[data_config["label_column"]].astype(float).unique().tolist())
        print(f"{split}: {path} | rows={len(frame)} | labels={labels}")

    print("\n=== 特征缓存 ===")
    expected_keys = {
        "protein": collect_unique_entities(
            data_config,
            data_config.get("protein_id_column"),
            data_config["sequence_column"],
            "protein",
        )[0],
        "drug": collect_unique_entities(
            data_config,
            data_config.get("drug_id_column"),
            data_config["smiles_column"],
            "drug",
        )[0],
    }
    missing_features: list[Path] = []
    for modality in ("protein", "drug"):
        for name, value in data_config["feature_files"][modality].items():
            path = Path(value)
            if path.is_file():
                store = FeatureStore.load(path, expected_name=name)
                expected_dim = int(model_specs[name]["expected_dim"])
                if store.dimension != expected_dim:
                    raise ValueError(
                        f"{path} 维度为 {store.dimension}，但配置期望 {expected_dim}。"
                    )
                expected = set(expected_keys[modality])
                actual = set(store.keys)
                if actual != expected:
                    raise ValueError(
                        f"{path} 实体集合与数据不一致："
                        f"missing={len(expected - actual)}，extra={len(actual - expected)}。"
                    )
                signature, _ = local_checkpoint_signature(
                    model_specs[name]["model_path"],
                    bool(model_specs[name].get("use_safetensors", True)),
                )
                expected_metadata = {
                    "source_id": model_specs[name]["source_id"],
                    "checkpoint_signature": signature,
                    "max_length": int(model_specs[name]["max_length"]),
                }
                if "dataset" in store.metadata:
                    expected_metadata["dataset"] = config["experiment"]["dataset"]
                if "inference_dtype" in store.metadata:
                    expected_metadata["inference_dtype"] = (
                        "float32"
                        if name == "molformer"
                        else str(
                            model_specs[name].get(
                                "dtype", config["feature_extraction"]["dtype"]
                            )
                        ).lower()
                    )
                mismatch = {
                    key: (store.metadata.get(key), expected_value)
                    for key, expected_value in expected_metadata.items()
                    if store.metadata.get(key) != expected_value
                }
                if mismatch:
                    raise ValueError(f"{path} 元数据与配置不一致：{mismatch}")
                print(f"{modality}/{name}: {path} | shape={tuple(store.embeddings.shape)}")
            else:
                missing_features.append(path)
                print(f"{modality}/{name}: 缺失 | {path}")
    if args.require_features and missing_features:
        raise FileNotFoundError(f"缺少 {len(missing_features)} 个特征文件。")
    print("\n检查完成。")


if __name__ == "__main__":
    main()
