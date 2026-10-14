from __future__ import annotations

import argparse
import gc
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

# 即使用户绕过 scripts/run_extract_hf.sh 直接执行 Python 入口，也绝不访问 Hub。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from ..config import DATASET_NAMES, load_config, normalize_dataset_name, resolved_config
from ..data import validate_fixed_split
from ..feature_store import FeatureStore, save_feature_store
from ..utils import resolve_device, set_global_seed
from .common import (
    biological_token_mask,
    batches,
    collect_unique_entities,
    local_checkpoint_signature,
    masked_mean,
    parse_torch_dtype,
)


LOGGER = logging.getLogger("molprotx.extract_hf")
LOCAL_MODEL_NAMES = ("esm2", "prott5", "ankh", "molformer", "chemberta", "moldeberta")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从六个本地预训练模型离线提取并缓存特征。")
    parser.add_argument("--config", default="configs/full.yaml", help="主配置文件路径。")
    parser.add_argument(
        "--dataset",
        required=True,
        type=normalize_dataset_name,
        choices=DATASET_NAMES,
        help="需要提取特征的数据集。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=LOCAL_MODEL_NAMES,
        default=list(LOCAL_MODEL_NAMES),
        help="需要提取的模型；默认依次提取全部六个本地模型。",
    )
    parser.add_argument("--device", default=None, help="覆盖配置中的设备，例如 cuda、cuda:1。")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖所有模型的 batch size。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的特征文件。")
    return parser


def _load_model_and_tokenizer(name: str, spec: dict[str, Any], dtype: torch.dtype, device: torch.device):
    model_path = spec["model_path"]
    if name == "molformer":
        modeling_file = Path(model_path) / "modeling_molformer.py"
        if not modeling_file.is_file():
            raise FileNotFoundError(f"MolFormer 缺少本地模型代码：{modeling_file}")
        if "transformers.masking_utils" in modeling_file.read_text(encoding="utf-8"):
            raise RuntimeError(
                "当前 MolFormer 模型代码来自 Transformers v5/main 分支，"
                "请替换为官方 compat-v4 分支文件后再运行。"
            )
        if dtype != torch.float32:
            LOGGER.warning("MolFormer 不支持稳定的 FP16 特征提取，已强制改用 FP32。")
            dtype = torch.float32
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(spec.get("trust_remote_code", False)),
        "local_files_only": True,
    }
    if name == "prott5":
        # ProtT5 的旧 SentencePiece Unigram 模型不能可靠地转换成 fast tokenizer；
        # Transformers 4.47.x 下显式使用 slow tokenizer 才能稳定加载。
        tokenizer = T5Tokenizer.from_pretrained(
            model_path, legacy=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **common_kwargs)
    load_dtype = torch.float32 if name == "molformer" else dtype
    model_kwargs = {
        **common_kwargs,
        "torch_dtype": load_dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": bool(spec.get("use_safetensors", True)),
    }
    # MLM checkpoint 中没有 pooler 权重，而 AutoModel 默认会为 ESM/Roberta 新建 pooler。
    # Transformers 4.47.1 配合 low_cpu_mem_usage 时，这些缺失参数可能残留在 meta device。
    if name in {"esm2", "chemberta"}:
        model_kwargs["add_pooling_layer"] = False
    if name == "molformer":
        model_kwargs["deterministic_eval"] = True

    def load_model(kwargs: dict[str, Any]) -> torch.nn.Module:
        if name in {"prott5", "ankh"}:
            return T5EncoderModel.from_pretrained(model_path, **kwargs)
        return AutoModel.from_pretrained(model_path, **kwargs)

    model = load_model(model_kwargs)
    meta_tensors = [
        tensor_name
        for tensor_name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        LOGGER.warning(
            "%s 使用 low_cpu_mem_usage 加载后仍有 %d 个 meta tensors（例如 %s），"
            "正在用标准 CPU 加载方式安全重试。",
            name,
            len(meta_tensors),
            ", ".join(meta_tensors[:3]),
        )
        del model
        gc.collect()
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs["low_cpu_mem_usage"] = False
        model = load_model(fallback_kwargs)
        remaining_meta = [
            tensor_name
            for tensor_name, tensor in (*model.named_parameters(), *model.named_buffers())
            if tensor.device.type == "meta"
        ]
        if remaining_meta:
            raise RuntimeError(
                f"{name} 标准加载后仍包含 meta tensors：{remaining_meta[:5]}。"
                "请检查 Transformers/PyTorch 版本和 checkpoint 完整性。"
            )
    if name == "molformer":
        # compat-v4 的线性注意力特征映射在 __init__ 中执行 CPU QR；CPU 不支持 FP16 geqrf。
        # 因此构造、加载和 GPU 前向均保持 FP32，输出向量再按 output_dtype 压缩保存。
        model.to(device=device, dtype=dtype)
    else:
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _prepare_texts(name: str, values: list[str]) -> tuple[list[Any], dict[str, Any]]:
    if name == "prott5":
        formatted = [" ".join(list(re.sub(r"[UZOB]", "X", sequence.upper()))) for sequence in values]
        return formatted, {}
    if name == "ankh":
        return [list(sequence.upper()) for sequence in values], {"is_split_into_words": True}
    if name == "molformer":
        # 部分 tokenizer.json 会默认返回 token_type_ids，但 MolformerModel.forward 不接收该参数。
        return values, {"return_token_type_ids": False}
    return values, {}


def _encode_batch(
    name: str,
    values: list[str],
    model: torch.nn.Module,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    texts, extra_tokenizer_kwargs = _prepare_texts(name, values)
    untruncated = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        **extra_tokenizer_kwargs,
    )
    truncated_count = sum(len(token_ids) > max_length for token_ids in untruncated["input_ids"])
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
        **extra_tokenizer_kwargs,
    )
    if name == "molformer":
        tokenized.pop("token_type_ids", None)
    tokenized = {key: value.to(device, non_blocking=True) for key, value in tokenized.items()}
    use_autocast = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype if use_autocast else torch.float32, enabled=use_autocast
    ):
        outputs = model(**tokenized)
        if name == "molformer" and getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = biological_token_mask(tokenized, list(tokenizer.all_special_ids))
            pooled = masked_mean(hidden, mask)
    pooled = pooled.detach().float().cpu()
    if not torch.isfinite(pooled).all():
        raise FloatingPointError(
            f"{name} 编码结果包含 NaN 或 Inf。请检查该模型的 dtype 与 GPU 混合精度支持。"
        )
    return pooled, truncated_count


def _extract_one(
    name: str,
    config: dict[str, Any],
    device: torch.device,
    batch_size_override: int | None,
    overwrite: bool,
) -> None:
    feature_config = config["feature_extraction"]
    data_config = config["data"]
    modality = "protein" if name in {"esm2", "prott5", "ankh"} else "drug"
    output_path = Path(data_config["feature_files"][modality][name])
    id_column = data_config["protein_id_column"] if modality == "protein" else data_config["drug_id_column"]
    value_column = data_config["sequence_column"] if modality == "protein" else data_config["smiles_column"]
    keys, values = collect_unique_entities(data_config, id_column, value_column, modality)
    spec = feature_config["models"][name]
    configured_inference_dtype = (
        "float32" if name == "molformer" else str(spec.get("dtype", feature_config["dtype"])).lower()
    )
    checkpoint_signature, weight_files = local_checkpoint_signature(
        spec["model_path"], bool(spec.get("use_safetensors", True))
    )
    if output_path.exists() and not overwrite:
        store = FeatureStore.load(output_path, expected_name=name)
        expected_dim = int(spec["expected_dim"])
        if store.dimension != expected_dim:
            raise ValueError(
                f"已有 {name} 缓存维度为 {store.dimension}，期望 {expected_dim}；"
                "核对后使用 --overwrite 重新提取。"
            )
        if len(store.keys) != len(keys) or set(store.keys) != set(keys):
            raise ValueError(f"已有 {name} 缓存实体与当前三个数据划分不一致。")
        expected_metadata = {
            "source_id": spec["source_id"],
            "checkpoint_signature": checkpoint_signature,
            "max_length": int(spec["max_length"]),
        }
        if "dataset" in store.metadata:
            expected_metadata["dataset"] = config["experiment"]["dataset"]
        if "inference_dtype" in store.metadata:
            expected_metadata["inference_dtype"] = configured_inference_dtype
        mismatched = {
            key: (store.metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if store.metadata.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f"已有 {name} 缓存的本地 checkpoint/max_length 与配置不一致：{mismatched}。"
                "核对后使用 --overwrite 重新提取。"
            )
        LOGGER.info("跳过 %s：特征已存在且通过校验，形状=%s", name, tuple(store.embeddings.shape))
        return
    batch_size = batch_size_override or int(
        feature_config["protein_batch_size"] if modality == "protein" else feature_config["drug_batch_size"]
    )
    dtype = parse_torch_dtype(spec.get("dtype", feature_config["dtype"]), device)
    output_dtype = parse_torch_dtype(feature_config.get("output_dtype", "float16"), torch.device("cuda"))
    LOGGER.info(
        "加载 %s（%s），实体数=%d，batch_size=%d，max_length=%d",
        name,
        spec["model_path"],
        len(keys),
        batch_size,
        int(spec["max_length"]),
    )
    model, tokenizer = _load_model_and_tokenizer(name, spec, dtype, device)
    chunks: list[torch.Tensor] = []
    total_truncated = 0
    progress = tqdm(total=len(values), desc=f"提取 {name}", unit="entity", dynamic_ncols=True)
    for batch_values in batches(values, batch_size):
        encoded, truncated_count = _encode_batch(
            name,
            batch_values,
            model,
            tokenizer,
            int(spec["max_length"]),
            device,
            dtype,
        )
        chunks.append(encoded)
        total_truncated += truncated_count
        progress.update(len(batch_values))
    progress.close()
    embeddings = torch.cat(chunks, dim=0).to(output_dtype)
    expected_dim = int(spec["expected_dim"])
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"{name} 输出维度为 {embeddings.shape[1]}，配置期望 {expected_dim}。"
            "请确认本地 checkpoint 与配置一致。"
        )
    metadata = {
        "dataset": config["experiment"]["dataset"],
        "modality": modality,
        "source_id": spec["source_id"],
        "source_revision": spec.get("source_revision"),
        "checkpoint_signature": checkpoint_signature,
        "weight_files": weight_files,
        "offline": True,
        "inference_dtype": configured_inference_dtype,
        "max_length": int(spec["max_length"]),
        "pooling": "model_pooler" if name == "molformer" else "masked_mean",
        "truncated_entities": total_truncated,
        "source_column": value_column,
        "dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
    }
    save_feature_store(
        output_path,
        name=name,
        keys=keys,
        embeddings=embeddings,
        metadata=metadata,
    )
    LOGGER.info(
        "已保存 %s：%s，形状=%s，达到 max_length 而截断的实体数=%d",
        name,
        output_path,
        tuple(embeddings.shape),
        total_truncated,
    )
    del model, tokenizer, embeddings, chunks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config, project_root = load_config(args.config, dataset=args.dataset)
    config = resolved_config(config, project_root)
    validate_fixed_split(config["data"])
    set_global_seed(int(config["experiment"]["seed"]))
    device = resolve_device(args.device or config["feature_extraction"]["device"])
    for name in args.models:
        _extract_one(
            name,
            config,
            device,
            args.batch_size,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
