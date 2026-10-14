from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .entities import entity_keys_from_frame
from .feature_store import FeatureStore
from .utils import make_generator, seed_worker


PROTEIN_MODELS = ("esm2", "prott5", "ankh")
DRUG_MODELS = ("molformer", "chemberta", "moldeberta")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fixed_split(data_config: dict[str, Any]) -> dict[str, Any]:
    paths = {split: Path(data_config[f"{split}_csv"]) for split in ("train", "val", "test")}
    parents = {path.parent.resolve() for path in paths.values()}
    if len(parents) != 1:
        raise ValueError("train/val/test 必须位于同一个固定划分目录。")
    manifest_path = next(iter(parents)) / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少固定划分清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("split_seed", -1)) != 42:
        raise ValueError(f"{manifest_path} 的 split_seed 不是 42。")
    if manifest.get("split_ratios") != {"train": 0.7, "val": 0.1, "test": 0.2}:
        raise ValueError(f"{manifest_path} 的 split_ratios 不是 7:1:2。")
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = manifest.get("splits", {}).get(split, {}).get("sha256")
        if not expected_hash or _sha256(path) != expected_hash:
            raise ValueError(f"{path} 与 split_manifest.json 中的 SHA-256 不一致。")
    return manifest


@dataclass(frozen=True)
class LoadedStores:
    protein: dict[str, FeatureStore]
    drug: dict[str, FeatureStore]

    @property
    def input_dimensions(self) -> dict[str, dict[str, int]]:
        return {
            "protein": {name: store.dimension for name, store in self.protein.items()},
            "drug": {name: store.dimension for name, store in self.drug.items()},
        }


def load_feature_stores(feature_files: dict[str, dict[str, str]]) -> LoadedStores:
    for modality, allowed in (("protein", PROTEIN_MODELS), ("drug", DRUG_MODELS)):
        names = set(feature_files.get(modality, {}))
        # Cache loading also serves single-view workbook ablations (A1/A3).
        # Architecture-specific view counts are validated by the model/config.
        if not names or not names <= set(allowed):
            raise ValueError(f"{modality} 特征缓存必须包含至少一个合法视角，且不能包含未知视角。")
    protein = {
        name: FeatureStore.load(feature_files["protein"][name], expected_name=name)
        for name in PROTEIN_MODELS if name in feature_files["protein"]
    }
    drug = {
        name: FeatureStore.load(feature_files["drug"][name], expected_name=name)
        for name in DRUG_MODELS if name in feature_files["drug"]
    }
    return LoadedStores(protein=protein, drug=drug)


class InteractionFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, csv_path: str | Path, data_config: dict[str, Any], stores: LoadedStores, split: str):
        self.csv_path = Path(csv_path)
        self.split = split
        drug_id_column = data_config.get("drug_id_column")
        protein_id_column = data_config.get("protein_id_column")
        required = [data_config["label_column"]]
        required.append(drug_id_column or data_config["smiles_column"])
        required.append(protein_id_column or data_config["sequence_column"])
        required = list(dict.fromkeys(required))
        header = pd.read_csv(self.csv_path, nrows=0)
        missing = [column for column in required if column not in header.columns]
        if missing:
            raise ValueError(f"{self.csv_path} 缺少列：{missing}")
        # 训练只需 ID 与标签，不把长 SMILES/蛋白序列复制到 DataLoader workers。
        frame = pd.read_csv(self.csv_path, usecols=required)

        self.drug_ids = entity_keys_from_frame(
            frame, drug_id_column, data_config["smiles_column"], "drug"
        )
        self.protein_ids = entity_keys_from_frame(
            frame, protein_id_column, data_config["sequence_column"], "protein"
        )
        labels = frame[data_config["label_column"]].astype(float)
        if not labels.isin([0.0, 1.0]).all():
            invalid = sorted(labels[~labels.isin([0.0, 1.0])].unique().tolist())
            raise ValueError(f"{self.csv_path} 含非二分类标签：{invalid}")
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)
        self.sample_count = len(frame)

        self.protein_features = {name: store.embeddings for name, store in stores.protein.items()}
        self.drug_features = {name: store.embeddings for name, store in stores.drug.items()}
        self.protein_indices = {
            name: store.indices_for(self.protein_ids) for name, store in stores.protein.items()
        }
        self.drug_indices = {name: store.indices_for(self.drug_ids) for name, store in stores.drug.items()}

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "protein": {
                name: self.protein_features[name][self.protein_indices[name][index]]
                for name in self.protein_features
            },
            "drug": {
                name: self.drug_features[name][self.drug_indices[name][index]]
                for name in self.drug_features
            },
            "label": self.labels[index],
            "row_index": index,
            "drug_id": self.drug_ids[index],
            "protein_id": self.protein_ids[index],
            "split": self.split,
        }


def build_datasets(
    data_config: dict[str, Any], stores: LoadedStores
) -> dict[str, InteractionFeatureDataset]:
    validate_fixed_split(data_config)
    return {
        split: InteractionFeatureDataset(data_config[f"{split}_csv"], data_config, stores, split)
        for split in ("train", "val", "test")
    }


def build_dataloaders(
    datasets: dict[str, InteractionFeatureDataset], training_config: dict[str, Any], seed: int
) -> dict[str, DataLoader]:
    num_workers = int(training_config["num_workers"])
    persistent_workers = bool(training_config.get("persistent_workers", False)) and num_workers > 0
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(training_config.get("pin_memory", True)),
        "worker_init_fn": seed_worker,
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            generator=make_generator(seed),
            drop_last=False,
            persistent_workers=persistent_workers,
            **common,
        ),
        "train_eval": DataLoader(
            datasets["train"],
            batch_size=int(training_config["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(training_config["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(training_config["eval_batch_size"]),
            shuffle=False,
            drop_last=False,
            persistent_workers=False,
            **common,
        ),
    }
