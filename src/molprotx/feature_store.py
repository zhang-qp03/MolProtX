from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .utils import atomic_torch_save, torch_load


@dataclass(frozen=True)
class FeatureStore:
    name: str
    keys: tuple[str, ...]
    embeddings: torch.Tensor
    metadata: dict[str, Any]
    key_to_index: dict[str, int]

    @classmethod
    def load(cls, path: str | Path, expected_name: str | None = None) -> "FeatureStore":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"特征文件不存在：{source}")
        payload = torch_load(source, map_location="cpu")
        required = {"name", "keys", "embeddings", "metadata"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"特征文件格式错误：{source}")
        name = str(payload["name"])
        if expected_name is not None and name != expected_name:
            raise ValueError(f"特征名称不匹配：期望 {expected_name}，文件中为 {name}")
        keys = tuple(str(key) for key in payload["keys"])
        embeddings = payload["embeddings"]
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError(f"embeddings 必须是二维 Tensor：{source}")
        if embeddings.shape[0] != len(keys):
            raise ValueError(f"keys 数量与 embeddings 行数不一致：{source}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"特征文件包含重复 key：{source}")
        if not torch.isfinite(embeddings.float()).all():
            raise ValueError(f"特征文件包含 NaN 或 Inf：{source}")
        return cls(
            name=name,
            keys=keys,
            embeddings=embeddings.contiguous(),
            metadata=dict(payload["metadata"]),
            key_to_index={key: index for index, key in enumerate(keys)},
        )

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def indices_for(self, keys: Iterable[str]) -> torch.Tensor:
        indices: list[int] = []
        missing: list[str] = []
        for key in keys:
            normalized = str(key)
            if normalized not in self.key_to_index:
                missing.append(normalized)
            else:
                indices.append(self.key_to_index[normalized])
        if missing:
            preview = ", ".join(missing[:5])
            raise KeyError(f"{self.name} 缺少 {len(missing)} 个实体特征，例如：{preview}")
        return torch.tensor(indices, dtype=torch.long)


def save_feature_store(
    path: str | Path,
    name: str,
    keys: list[str],
    embeddings: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(keys):
        raise ValueError("保存特征时 keys 与 embeddings 形状不一致。")
    if len(set(keys)) != len(keys):
        raise ValueError("保存特征时检测到重复 key。")
    if not torch.isfinite(embeddings.detach().float()).all():
        raise ValueError("保存特征时检测到 NaN 或 Inf。")
    payload = {
        "format_version": 1,
        "name": name,
        "keys": list(keys),
        "embeddings": embeddings.detach().cpu().contiguous(),
        "metadata": metadata,
    }
    atomic_torch_save(payload, path)
