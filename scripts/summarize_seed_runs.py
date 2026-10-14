from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from molprotx.config import (
    DATASET_NAMES,
    load_config,
    normalize_dataset_name,
    resolve_path,
    seeded_output_dir,
)


DEFAULT_SEEDS = (0, 1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 42, 43, 47, 53, 59, 67, 71, 73, 97)
SPLITS = ("train", "val", "test")
METRICS = ("auc", "auprc", "mcc", "accuracy", "precision", "recall", "specificity", "f1_score")
DISPLAY_NAMES = {
    "auc": "AUC",
    "auprc": "AUPRC",
    "mcc": "MCC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1_score": "F1-score",
}
LEGACY_NAMES = {"auc": "auroc", "f1_score": "f1"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总多个随机种子的 MolProtX 主实验结果。")
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument(
        "--dataset", required=True, type=normalize_dataset_name, choices=DATASET_NAMES
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-stem", default="summary_20seeds")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许部分 seed 尚未完成，并生成阶段性 summary。",
    )
    return parser


def _metric_value(values: dict[str, Any], name: str, path: Path) -> float:
    key = name if name in values else LEGACY_NAMES.get(name)
    if key is None or key not in values:
        raise KeyError(f"{path} 缺少指标 {name}")
    value = float(values[key])
    if not math.isfinite(value):
        raise ValueError(f"{path} 的指标 {name} 不是有限数值：{value}")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def collect_results(
    config_path: str, seeds: list[int], dataset: str = "biosnap"
) -> tuple[pd.DataFrame, list[int], Path]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("随机种子列表包含重复值。")
    config, project_root = load_config(config_path, dataset=dataset)
    dataset = str(config["experiment"]["dataset"])
    configured_output = config["experiment"]["output_dir"]
    summary_root = resolve_path(project_root, seeded_output_dir(configured_output, seeds[0]).parent)
    rows: list[dict[str, float | int | str]] = []
    missing: list[int] = []
    for seed in seeds:
        run_dir = resolve_path(project_root, seeded_output_dir(configured_output, seed))
        metrics_path = run_dir / "metrics_final.json"
        if not metrics_path.is_file():
            missing.append(seed)
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split in SPLITS:
            if split not in payload or not isinstance(payload[split], dict):
                raise ValueError(f"{metrics_path} 缺少 {split} 指标。")
            row: dict[str, float | int | str] = {
                "seed": seed,
                "dataset": dataset,
                "split": split,
                "output_dir": str(run_dir),
            }
            for metric in METRICS:
                row[metric] = _metric_value(payload[split], metric, metrics_path)
            rows.append(row)
    return pd.DataFrame(rows), missing, summary_root


def build_markdown(
    frame: pd.DataFrame,
    seeds: list[int],
    missing: list[int],
    dataset: str | None = None,
) -> str:
    completed = sorted(frame["seed"].unique().tolist()) if not frame.empty else []
    dataset_name = dataset or (str(frame.iloc[0]["dataset"]) if not frame.empty else "unknown")
    lines = [
        f"# MolProtX {dataset_name}：{len(seeds)} 个随机种子实验汇总",
        "",
        f"- 预注册随机种子：`{', '.join(map(str, seeds))}`",
        f"- 已完成：**{len(completed)}/{len(seeds)}**",
        f"- 未完成：`{', '.join(map(str, missing)) if missing else '无'}`",
        "- 最佳 seed 选择规则：验证集 AUPRC 最大；若并列，则验证集 AUC 最大，再按 seed 从小到大。",
        "- 标准差：跨 seed 的样本标准差（sample SD，ddof=1）。",
        "",
    ]
    if frame.empty:
        lines.extend(["尚无已完成实验。", ""])
        return "\n".join(lines)

    val_rows = frame[frame["split"] == "val"].sort_values(
        ["auprc", "auc", "seed"], ascending=[False, False, True]
    )
    best_seed = int(val_rows.iloc[0]["seed"])
    best = frame[frame["seed"] == best_seed].set_index("split")
    best_weights = Path(str(best.loc["val", "output_dir"])) / "checkpoints" / "best_model_weights.pt"
    lines.extend(
        [
            f"## 最佳随机种子：{best_seed}",
            "",
            f"最佳权重：`{best_weights}`",
            "",
            "| 指标 | Train | Validation | Test |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        lines.append(
            f"| {DISPLAY_NAMES[metric]} | {best.loc['train', metric]:.6f} | "
            f"{best.loc['val', metric]:.6f} | {best.loc['test', metric]:.6f} |"
        )

    lines.extend(
        [
            "",
            f"## 跨随机种子的 Mean ± SD（n={len(completed)}）",
            "",
            "| 指标 | Train | Validation | Test |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        formatted: list[str] = []
        for split in SPLITS:
            values = frame[frame["split"] == split][metric]
            standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            formatted.append(f"{values.mean():.6f} ± {standard_deviation:.6f}")
        lines.append(
            f"| {DISPLAY_NAMES[metric]} | {formatted[0]} | {formatted[1]} | {formatted[2]} |"
        )

    lines.extend(
        [
            "",
            "## 验证集排名",
            "",
            "| 排名 | Seed | Validation AUPRC | Validation AUC | Test AUPRC | Test AUC |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    test_by_seed = frame[frame["split"] == "test"].set_index("seed")
    for rank, (_, row) in enumerate(val_rows.iterrows(), start=1):
        seed = int(row["seed"])
        lines.append(
            f"| {rank} | {seed} | {row['auprc']:.6f} | {row['auc']:.6f} | "
            f"{test_by_seed.loc[seed, 'auprc']:.6f} | {test_by_seed.loc[seed, 'auc']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    seeds = [int(seed) for seed in args.seeds]
    output_stem = str(args.output_stem).strip()
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("--output-stem 必须是不含目录的文件名主体。")
    frame, missing, summary_root = collect_results(args.config, seeds, dataset=args.dataset)
    if missing and not args.allow_incomplete:
        raise FileNotFoundError(f"以下 seed 尚未产生 metrics_final.json：{missing}")
    summary_path = summary_root / f"{output_stem}.md"
    raw_path = summary_root / f"{output_stem}.csv"
    _atomic_write_text(
        summary_path,
        build_markdown(frame, seeds, missing, dataset=args.dataset),
    )
    if not frame.empty:
        temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
        frame.sort_values(["seed", "split"]).to_csv(temporary, index=False)
        temporary.replace(raw_path)
    print(f"已写入汇总：{summary_path}")
    if not frame.empty:
        print(f"已写入逐 seed 指标：{raw_path}")


if __name__ == "__main__":
    main()
