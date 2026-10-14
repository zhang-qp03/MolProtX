from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# 六种 PLM 特征始终从 pretrained/ 与本地缓存读取，禁止训练进程访问 Hugging Face Hub。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

from .config import (
    DATASET_NAMES,
    load_config,
    normalize_dataset_name,
    resolved_config,
    save_yaml,
    seeded_output_dir,
)
from .data import build_dataloaders, build_datasets, load_feature_stores, validate_fixed_split
from .model import MolProtXModel, training_model_class
from .training import ExperimentTrainer
from .utils import (
    configure_logging,
    count_parameters,
    prepare_output_dir,
    resolve_device,
    set_global_seed,
    write_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 MolProtX 药物-蛋白结合分类模型。")
    parser.add_argument("--config", default="configs/full.yaml", help="最终模型配置文件。")
    parser.add_argument(
        "--dataset",
        required=True,
        type=normalize_dataset_name,
        choices=DATASET_NAMES,
        help="训练数据集名称。",
    )
    parser.add_argument("--device", default=None, help="覆盖配置中的训练设备，例如 cuda:1。")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子并自动使用独立输出目录。")
    parser.add_argument("--resume", default=None, help="从 last_checkpoint.pt 恢复训练。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖既有输出目录。")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config, project_root = load_config(args.config, dataset=args.dataset)
    model_class = training_model_class(config["model"])
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
        config["experiment"]["output_dir"] = str(
            seeded_output_dir(config["experiment"]["output_dir"], args.seed)
        )
    config = resolved_config(config, project_root)
    if args.device:
        config["training"]["device"] = args.device

    seed = int(config["experiment"]["seed"])
    dataset = str(config["experiment"]["dataset"])
    set_global_seed(seed)
    output_dir = Path(config["experiment"]["output_dir"])
    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = prepare_output_dir(output_dir, overwrite=args.overwrite)
    logger = configure_logging(output_dir, level=logging.INFO)
    logger.info("MolProtX 模型启动，数据集=%s，固定随机种子=%d", dataset, seed)
    logger.info("模型=%s；%s；dropout-low + EMA + warm-up", model_class.__name__,
                "原 Full：双向跨模态注意力 + 拼接后池化 + 显式交互，分类器输入 224 维"
                if model_class is MolProtXModel else
                "A9v2 独立实验：两侧分别池化，无显式交互，分类器输入 64 维")
    logger.info("项目根目录：%s", project_root)
    save_yaml(config, output_dir / "config_resolved.yaml")
    split_manifest = validate_fixed_split(config["data"])
    logger.info(
        "固定数据划分已校验：seed=%s，ratio=%s",
        split_manifest["split_seed"],
        split_manifest["split_ratios"],
    )

    device = resolve_device(config["training"]["device"])
    if device.type == "cuda":
        logger.info("训练设备：%s | GPU=%s", device, torch.cuda.get_device_name(device))
    else:
        logger.info("训练设备：%s", device)

    feature_directories = sorted(
        {
            str(Path(path).parent)
            for modality in config["data"]["feature_files"].values()
            for path in modality.values()
        }
    )
    logger.info("加载数据集 %s 的本地缓存：%s", dataset, feature_directories)
    stores = load_feature_stores(config["data"]["feature_files"])
    input_dims = stores.input_dimensions
    logger.info("输入特征维度：%s", input_dims)
    datasets = build_datasets(config["data"], stores)
    logger.info(
        "数据规模：train=%d，val=%d，test=%d",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
    )
    loaders = build_dataloaders(datasets, config["training"], seed)
    model = model_class(input_dims=input_dims, config=config["model"])
    parameter_counts = count_parameters(model)
    logger.info(
        "融合与分类模型参数：total=%s，trainable=%s",
        f"{parameter_counts['total']:,}",
        f"{parameter_counts['trainable']:,}",
    )
    write_json(
        {"input_dims": input_dims, "parameters": parameter_counts, "model": str(model)},
        output_dir / "model_summary.json",
    )

    trainer = ExperimentTrainer(
        model=model,
        loaders=loaders,
        config=config,
        input_dims=input_dims,
        output_dir=output_dir,
        device=device,
        logger=logger,
    )
    results = trainer.fit(resume_path=args.resume)
    logger.info("数据集 %s 训练完成。最终结果：%s", dataset, results)


if __name__ == "__main__":
    main()
