from __future__ import annotations

import hashlib

import pandas as pd


def entity_keys_from_frame(
    frame: pd.DataFrame,
    id_column: str | None,
    value_column: str,
    entity_type: str,
) -> list[str]:
    """读取显式实体 ID；缺少 ID 列时根据原始实体内容生成稳定键。"""
    if id_column is not None:
        if id_column not in frame.columns:
            raise ValueError(f"缺少实体 ID 列：{id_column}")
        identifiers = frame[id_column]
        if identifiers.isna().any():
            raise ValueError(f"{id_column} 包含空值。")
        return identifiers.astype(str).tolist()

    if value_column not in frame.columns:
        raise ValueError(f"缺少实体内容列：{value_column}")
    values = frame[value_column]
    if values.isna().any():
        raise ValueError(f"{value_column} 包含空值，无法生成实体键。")

    prefix = str(entity_type).strip().lower()
    if not prefix:
        raise ValueError("entity_type 不能为空。")
    return [
        f"{prefix}_{hashlib.sha256(str(value).encode('utf-8')).hexdigest()}"
        for value in values
    ]
