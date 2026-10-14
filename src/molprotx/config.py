from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置文件不完整或不合法。"""


DATASET_NAMES = (
    "bindingdb",
    "biosnap",
    "celegans",
    "davis",
    "drugbank",
    "human",
    "kiba",
    "kinase",
)


def normalize_dataset_name(value: str) -> str:
    name = str(value).strip().lower()
    aliases = {
        "biosnap_full": "biosnap",
        "binding_db": "bindingdb",
        "c_elegans": "celegans",
        "c. elegans": "celegans",
    }
    name = aliases.get(name, name)
    if name not in DATASET_NAMES:
        raise ConfigError(f"不支持的数据集 {value!r}；可选值为 {', '.join(DATASET_NAMES)}。")
    return name


def select_dataset(config: dict[str, Any], dataset: str | None) -> dict[str, Any]:
    result = deepcopy(config)
    requested = normalize_dataset_name(dataset or result.get("experiment", {}).get("dataset", ""))
    registry = result.pop("datasets", None)
    if not isinstance(registry, dict) or requested not in registry:
        raise ConfigError(f"配置缺少 datasets.{requested}。")
    result["data"] = deepcopy(registry[requested])
    result["experiment"]["dataset"] = requested
    for key in ("name", "output_dir"):
        value = str(result["experiment"].get(key, ""))
        result["experiment"][key] = value.format(dataset=requested)
    return result


def load_config(path: str | Path, dataset: str | None = None) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ConfigError("YAML 顶层必须是映射。")

    project_root = config_path.parent.parent
    config = select_dataset(config, dataset)
    validate_config(config)
    return config, project_root


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "model", "training")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ConfigError(f"缺少配置节：{', '.join(missing)}")

    feature_files = config["data"].get("feature_files", {})
    expected = {
        "protein": {"esm2", "prott5", "ankh"},
        "drug": {"molformer", "chemberta", "moldeberta"},
    }
    active = config["model"].get("active_models")
    if active is not None:
        if set(active) != set(expected):
            raise ConfigError("model.active_models 必须恰好包含 protein 和 drug。")
        for modality, allowed in expected.items():
            names = active.get(modality, [])
            if len(names) < 2 or len(set(names)) != len(names) or not set(names) <= allowed:
                raise ConfigError(f"model.active_models.{modality} 必须包含至少两个合法的不重复视角。")
        expected = {modality: set(names) for modality, names in active.items()}
    for modality, names in expected.items():
        actual = set(feature_files.get(modality, {}))
        if actual != names:
            raise ConfigError(
                f"data.feature_files.{modality} 必须恰好包含 {sorted(names)}，当前为 {sorted(actual)}"
            )

    training = config["training"]
    if training.get("monitor_mode") not in {"min", "max"}:
        raise ConfigError("training.monitor_mode 必须是 min 或 max。")
    if training.get("mixed_precision") not in {"no", "fp16", "bf16"}:
        raise ConfigError("training.mixed_precision 必须是 no、fp16 或 bf16。")
    threshold = float(training.get("decision_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ConfigError("training.decision_threshold 必须位于 (0, 1)。")
    for key in ("max_epochs", "batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        if int(training.get(key, 0)) <= 0:
            raise ConfigError(f"training.{key} 必须大于 0。")
    if float(training.get("gradient_clip_norm", 0.0)) <= 0.0:
        raise ConfigError("training.gradient_clip_norm 必须大于 0。")

    ema = training.get("ema", {})
    if ema.get("enabled", False) and not 0.0 < float(ema.get("decay", 0.0)) < 1.0:
        raise ConfigError("training.ema.decay 必须位于 (0, 1)。")
    warmup = training.get("warmup", {})
    if warmup.get("enabled", False):
        if int(warmup.get("epochs", 0)) <= 0:
            raise ConfigError("training.warmup.epochs 必须大于 0。")
        if not 0.0 < float(warmup.get("start_factor", 0.0)) <= 1.0:
            raise ConfigError("training.warmup.start_factor 必须位于 (0, 1]。")

    # Historical configs may still describe the old explicit-interaction model.
    if "explicit_interaction" in config["model"]:
        interaction = config["model"]["explicit_interaction"]
        for key in ("bilinear_rank", "bilinear_output_dim"):
            if int(interaction.get(key, 0)) <= 0:
                raise ConfigError(f"model.explicit_interaction.{key} 必须大于 0。")
        if not 0.0 <= float(interaction.get("dropout", -1.0)) < 1.0:
            raise ConfigError("model.explicit_interaction.dropout 必须位于 [0, 1)。")

    model_specs = config.get("feature_extraction", {}).get("models", {})
    expected_models = expected["protein"] | expected["drug"]
    missing_specs = expected_models - set(model_specs)
    if missing_specs:
        raise ConfigError(f"feature_extraction.models 缺少：{sorted(missing_specs)}")
    for name in expected_models:
        if int(model_specs[name].get("expected_dim", 0)) <= 0:
            raise ConfigError(f"feature_extraction.models.{name}.expected_dim 必须大于 0。")
        if not str(model_specs[name].get("model_path", "")).strip():
            raise ConfigError(f"feature_extraction.models.{name}.model_path 不能为空。")
        if not str(model_specs[name].get("source_id", "")).strip():
            raise ConfigError(f"feature_extraction.models.{name}.source_id 不能为空。")
        if int(model_specs[name].get("max_length", 0)) <= 0:
            raise ConfigError(f"feature_extraction.models.{name}.max_length 必须大于 0。")


def resolved_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result = deepcopy(config)
    data = result["data"]
    for key in ("train_csv", "val_csv", "test_csv"):
        data[key] = str(resolve_path(project_root, data[key]))
    for modality in ("protein", "drug"):
        for name, value in data["feature_files"][modality].items():
            data["feature_files"][modality][name] = str(resolve_path(project_root, value))
    for spec in result["feature_extraction"]["models"].values():
        spec["model_path"] = str(resolve_path(project_root, spec["model_path"]))
    result["experiment"]["output_dir"] = str(
        resolve_path(project_root, result["experiment"]["output_dir"])
    )
    return result


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def seeded_output_dir(value: str | Path, seed: int) -> Path:
    """把主配置的输出目录稳定映射到指定 seed，避免不同重复实验相互覆盖。"""
    text = str(value)
    if "{seed}" in text:
        return Path(text.format(seed=int(seed)))
    path = Path(text)
    if re.fullmatch(r"seed-?\d+", path.name, flags=re.IGNORECASE):
        return path.with_name(f"seed{int(seed)}")
    replaced, count = re.subn(r"_seed-?\d+$", f"_seed{int(seed)}", text)
    return Path(replaced if count else f"{text}_seed{int(seed)}")


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
