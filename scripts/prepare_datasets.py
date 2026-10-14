from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
STANDARD_COLUMNS = ("Drug_ID", "Protein_ID", "SMILES", "Protein", "Label", "Source_Row")


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    reader: Callable[[Path], pd.DataFrame]


def _hashed_ids(values: pd.Series, prefix: str) -> pd.Series:
    return values.astype(str).map(
        lambda value: f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    )


def _standardize(
    frame: pd.DataFrame,
    *,
    smiles: str | int,
    protein: str | int,
    label: str | int,
    drug_id: str | int | None = None,
    protein_id: str | int | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "SMILES": frame[smiles].astype(str),
            "Protein": frame[protein].astype(str),
            "Label": pd.to_numeric(frame[label], errors="raise").astype(int),
            # Source_Row is zero-based and points to the original data row, excluding its header.
            "Source_Row": np.arange(len(frame), dtype=np.int64),
        }
    )
    result.insert(
        0,
        "Protein_ID",
        (
            frame[protein_id].astype(str)
            if protein_id is not None
            else _hashed_ids(result["Protein"], "protein")
        ),
    )
    result.insert(
        0,
        "Drug_ID",
        frame[drug_id].astype(str) if drug_id is not None else _hashed_ids(result["SMILES"], "drug"),
    )
    if result[["Drug_ID", "Protein_ID", "SMILES", "Protein", "Label"]].isna().any().any():
        raise ValueError("核心字段包含空值。")
    labels = set(result["Label"].unique().tolist())
    if not labels <= {0, 1} or labels != {0, 1}:
        raise ValueError(f"标签必须同时包含 0 和 1，实际为 {sorted(labels)}。")
    if (result["SMILES"].str.len() == 0).any() or (result["Protein"].str.len() == 0).any():
        raise ValueError("SMILES 或 Protein 包含空字符串。")
    return result.loc[:, STANDARD_COLUMNS]


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return _standardize(frame, smiles="SMILES", protein="Protein", label="Y")


def _read_three_column_txt(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 3:
        raise ValueError(f"{path} 应有 3 列，实际为 {frame.shape[1]} 列。")
    return _standardize(frame, smiles=0, protein=1, label=2)


def _read_five_column_txt(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 5:
        raise ValueError(f"{path} 应有 5 列，实际为 {frame.shape[1]} 列。")
    return _standardize(
        frame,
        drug_id=0,
        protein_id=1,
        smiles=2,
        protein=3,
        label=4,
    )


def _read_kinase(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(
        path,
        usecols=["UNIPROT_ID", "SEQUENCE", "Label", "DRUG_ID", "SMILES"],
        engine="openpyxl",
    )
    return _standardize(
        frame,
        drug_id="DRUG_ID",
        protein_id="UNIPROT_ID",
        smiles="SMILES",
        protein="SEQUENCE",
        label="Label",
    )


DATASETS = {
    "bindingdb": DatasetSpec("BindingDB.csv", _read_csv),
    "biosnap": DatasetSpec("BioSNAP.csv", _read_csv),
    "celegans": DatasetSpec("celegans.txt", _read_three_column_txt),
    "davis": DatasetSpec("Davis.txt", _read_five_column_txt),
    "drugbank": DatasetSpec("DrugBank.txt", _read_three_column_txt),
    "human": DatasetSpec("Human.csv", _read_csv),
    "kiba": DatasetSpec("KIBA.txt", _read_five_column_txt),
    "kinase": DatasetSpec("Kinase.xlsx", _read_kinase),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_sizes(sample_count: int) -> dict[str, int]:
    # Floors train/val and assigns the integer remainder to test. This is deterministic and
    # preserves every row while keeping the requested 7:1:2 ratio to within one sample.
    train_count = int(sample_count * SPLIT_RATIOS["train"])
    val_count = int(sample_count * SPLIT_RATIOS["val"])
    return {
        "train": train_count,
        "val": val_count,
        "test": sample_count - train_count - val_count,
    }


def prepare_dataset(
    dataset: str,
    source_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, object]:
    spec = DATASETS[dataset]
    source = source_root / spec.filename
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_root / dataset
    targets = {split: destination / f"{split}.csv" for split in SPLIT_RATIOS}
    manifest_path = destination / "split_manifest.json"
    existing = [path for path in [*targets.values(), manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{destination} 已有划分文件；如需按相同协议重建，请添加 --overwrite。"
        )

    frame = spec.reader(source)
    sizes = _split_sizes(len(frame))
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(frame))
    shuffled = frame.iloc[permutation].reset_index(drop=True)
    boundaries = {
        "train": (0, sizes["train"]),
        "val": (sizes["train"], sizes["train"] + sizes["val"]),
        "test": (sizes["train"] + sizes["val"], len(frame)),
    }

    destination.mkdir(parents=True, exist_ok=True)
    split_details: dict[str, dict[str, object]] = {}
    for split, (start, stop) in boundaries.items():
        subset = shuffled.iloc[start:stop]
        temporary = targets[split].with_suffix(".csv.tmp")
        subset.to_csv(temporary, index=False)
        temporary.replace(targets[split])
        split_details[split] = {
            "rows": len(subset),
            "label_counts": {
                str(label): int(count)
                for label, count in subset["Label"].value_counts().sort_index().items()
            },
            "sha256": _sha256(targets[split]),
        }

    source_rows = pd.concat(
        [pd.read_csv(targets[split], usecols=["Source_Row"]) for split in SPLIT_RATIOS],
        ignore_index=True,
    )["Source_Row"]
    if len(source_rows) != len(frame) or sorted(source_rows.tolist()) != list(range(len(frame))):
        raise RuntimeError(f"{dataset} 划分完整性检查失败。")
    split_sets = {
        split: set(pd.read_csv(path, usecols=["Source_Row"])["Source_Row"].tolist())
        for split, path in targets.items()
    }
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError(f"{dataset} 划分之间存在重叠。")

    manifest: dict[str, object] = {
        "dataset": dataset,
        "source_file": str(source.as_posix()),
        "source_sha256": _sha256(source),
        "source_rows": len(frame),
        "source_duplicate_rows": int(frame.duplicated(subset=["SMILES", "Protein", "Label"]).sum()),
        "columns": list(STANDARD_COLUMNS),
        "split_seed": SPLIT_SEED,
        "shuffle_algorithm": "numpy.random.default_rng(42).permutation",
        "split_ratios": SPLIT_RATIOS,
        "integer_rounding": "floor train, floor val, remainder test",
        "splits": split_details,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def check_dataset(dataset: str, source_root: Path, output_root: Path) -> dict[str, object]:
    spec = DATASETS[dataset]
    source = source_root / spec.filename
    destination = output_root / dataset
    manifest_path = destination / "split_manifest.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "dataset": dataset,
        "source_sha256": _sha256(source),
        "split_seed": SPLIT_SEED,
        "split_ratios": SPLIT_RATIOS,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_protocol.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{manifest_path} 与当前原始数据/划分协议不一致：{mismatches}")
    total = 0
    seen: set[int] = set()
    for split in SPLIT_RATIOS:
        path = destination / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = manifest["splits"][split]["sha256"]
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{path} 的 SHA-256 与 split_manifest.json 不一致。")
        rows = pd.read_csv(path, usecols=["Source_Row"])["Source_Row"].astype(int).tolist()
        if len(rows) != int(manifest["splits"][split]["rows"]):
            raise ValueError(f"{path} 的行数与 split_manifest.json 不一致。")
        if seen.intersection(rows):
            raise ValueError(f"{path} 与其他 split 存在 Source_Row 重叠。")
        seen.update(rows)
        total += len(rows)
    if total != int(manifest["source_rows"]) or seen != set(range(total)):
        raise ValueError(f"{dataset} 的固定划分没有完整覆盖原始行。")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 8 个原始 DTI 数据集按固定 seed=42 随机打乱并切分为 7:1:2。"
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--source-root", default="dataset")
    parser.add_argument("--output-root", default="dataset/splits")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check", action="store_true", help="只验证现有划分和哈希，不写文件。")
    return parser


def main() -> None:
    args = _parser().parse_args()
    requested = list(DATASETS) if args.dataset == "all" else [args.dataset]
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for dataset in requested:
        manifest = (
            check_dataset(dataset, source_root, output_root)
            if args.check
            else prepare_dataset(dataset, source_root, output_root, args.overwrite)
        )
        sizes = manifest["splits"]
        print(
            f"{dataset}: {'check=ok, ' if args.check else ''}"
            f"train={sizes['train']['rows']}, val={sizes['val']['rows']}, "
            f"test={sizes['test']['rows']} -> {output_root / dataset}"
        )


if __name__ == "__main__":
    main()
