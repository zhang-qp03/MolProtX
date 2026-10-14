from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..entities import entity_keys_from_frame


_CHECKPOINT_AUXILIARY_FILES = {
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _selected_weight_files(model_path: Path, use_safetensors: bool) -> list[Path]:
    stem = "model.safetensors" if use_safetensors else "pytorch_model.bin"
    single = model_path / stem
    index = model_path / f"{stem}.index.json"
    if single.is_file():
        return [single]
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        files = [model_path / name for name in names]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"权重索引引用了缺失分片：{missing}")
        return [index, *files]
    raise FileNotFoundError(
        f"{model_path} 缺少 {stem} 或 {stem}.index.json；"
        "请确认下载的是完整模型，而不是 Git LFS 指针。"
    )


def _sample_large_file(path: Path, digest: Any) -> None:
    """对大权重做固定位置抽样，快速发现空文件、LFS 指针或传输截断。"""
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    if size < 1024:
        head = path.read_bytes()
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"{path} 是 Git LFS 指针，不是真实权重文件。")
        digest.update(head)
        return
    chunk_size = min(65536, size)
    offsets = sorted({0, max(0, size // 2 - chunk_size // 2), max(0, size - chunk_size)})
    with path.open("rb") as file:
        for offset in offsets:
            file.seek(offset)
            digest.update(file.read(chunk_size))


def local_checkpoint_signature(model_path: str | Path, use_safetensors: bool) -> tuple[str, list[str]]:
    """生成可跨机器复核的本地 checkpoint 签名，不把绝对路径写入缓存。"""
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"本地预训练模型目录不存在：{root}")
    auxiliary = [
        path
        for path in root.iterdir()
        if path.is_file() and (path.name in _CHECKPOINT_AUXILIARY_FILES or path.suffix == ".py")
    ]
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"本地预训练模型缺少 config.json：{root}")
    weights = _selected_weight_files(root, use_safetensors)
    files = sorted({*auxiliary, *weights}, key=lambda path: path.name)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        if path in weights and path.suffix in {".bin", ".safetensors"}:
            _sample_large_file(path, digest)
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest(), [path.name for path in weights]


def collect_unique_entities(
    data_config: dict[str, Any], id_column: str | None, value_column: str, entity_type: str
) -> tuple[list[str], list[str]]:
    """按 train→val→test 的首次出现顺序收集实体，不改变任何数据划分或标签。"""
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(data_config[f"{split}_csv"])
        columns = [value_column] if id_column is None else [id_column, value_column]
        frame = pd.read_csv(path, usecols=columns)
        keys = entity_keys_from_frame(frame, id_column, value_column, entity_type)
        values = frame[value_column].astype(str).tolist()
        for key, value in zip(keys, values, strict=True):
            if key in mapping and mapping[key] != value:
                raise ValueError(f"实体 {key} 在不同记录中对应了不一致的 {value_column}。")
            mapping.setdefault(key, value)
    return list(mapping), list(mapping.values())


def batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_torch_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if device.type == "cpu":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"不支持的 dtype：{name}")
    return mapping[normalized]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def biological_token_mask(tokenized: dict[str, torch.Tensor], special_token_ids: list[int]) -> torch.Tensor:
    mask = tokenized["attention_mask"].bool()
    input_ids = tokenized["input_ids"]
    for token_id in special_token_ids:
        mask &= input_ids.ne(int(token_id))
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty] = tokenized["attention_mask"][empty].bool()
    return mask
