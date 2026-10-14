from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .metrics import binary_classification_metrics, format_metrics
from .utils import atomic_torch_save, torch_load, write_json


@dataclass
class EpochResult:
    metrics: dict[str, float | int]
    predictions: pd.DataFrame


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay 必须位于 (0, 1)。")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any], model: nn.Module) -> None:
        self.decay = float(state["decay"])
        parameters = dict(model.named_parameters())
        loaded = state["shadow"]
        if set(loaded) != set(self.shadow):
            raise ValueError("checkpoint 中的 EMA 参数集合与当前模型不一致。")
        self.shadow = {
            name: tensor.to(device=parameters[name].device, dtype=parameters[name].dtype)
            for name, tensor in loaded.items()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])

    @contextmanager
    def average_parameters(self, model: nn.Module):
        backup = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                parameters = dict(model.named_parameters())
                for name, value in backup.items():
                    parameters[name].copy_(value)


def _move_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in features.items()}


def _autocast_settings(precision: str, device: torch.device) -> tuple[bool, torch.dtype]:
    if precision == "fp16":
        return device.type == "cuda", torch.float16
    if precision == "bf16":
        return device.type == "cuda", torch.bfloat16
    return False, torch.float32


def _make_grad_scaler(enabled: bool) -> Any:
    """兼容 PyTorch 2.2--2.5，并优先使用未弃用的统一 AMP API。"""
    unified_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if unified_scaler is not None:
        return unified_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """将从 CPU checkpoint 读取的优化器张量迁移到训练设备。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    precision: str,
    split: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 1.0,
    collect_predictions: bool = True,
    ema: ExponentialMovingAverage | None = None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    autocast_enabled, autocast_dtype = _autocast_settings(precision, device)
    if training:
        optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    sample_count = 0
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    row_indices: list[int] = []
    drug_ids: list[str] = []
    protein_ids: list[str] = []
    description = f"{split} {'训练' if training else '评估'}"
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True, leave=False)

    for batch_index, batch in enumerate(progress):
        labels = batch["label"].to(device, non_blocking=True)
        protein = _move_features(batch["protein"], device)
        drug = _move_features(batch["drug"], device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(protein=protein, drug=drug)
            loss = criterion(logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError(
                f"{split} 的 batch {batch_index} 出现 NaN/Inf logits 或 loss；"
                "请检查特征缓存、学习率和混合精度设置。"
            )

        if training:
            total_batches = len(loader)
            final_group_size = total_batches % gradient_accumulation_steps
            if final_group_size == 0:
                final_group_size = gradient_accumulation_steps
            in_final_group = batch_index >= total_batches - final_group_size
            accumulation_divisor = final_group_size if in_final_group else gradient_accumulation_steps
            scaled_loss = loss / accumulation_divisor
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (batch_index + 1) % gradient_accumulation_steps == 0 or (
                batch_index + 1 == len(loader)
            )
            if should_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                )
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        probabilities = torch.sigmoid(logits.detach())
        labels_all.append(labels.detach().cpu().numpy())
        # NumPy 不支持 BF16；统一转为 FP32 后再保存指标与预测。
        logits_all.append(logits.detach().float().cpu().numpy())
        probabilities_all.append(probabilities.float().cpu().numpy())
        if collect_predictions:
            row_indices.extend(int(value) for value in batch["row_index"])
            drug_ids.extend(str(value) for value in batch["drug_id"])
            protein_ids.extend(str(value) for value in batch["protein_id"])
        progress.set_postfix(loss=f"{loss_sum / sample_count:.4f}")

    labels_np = np.concatenate(labels_all)
    logits_np = np.concatenate(logits_all)
    probabilities_np = np.concatenate(probabilities_all)
    mean_loss = loss_sum / max(sample_count, 1)
    metrics = binary_classification_metrics(labels_np, probabilities_np, threshold, loss=mean_loss)
    if collect_predictions:
        prediction_frame = pd.DataFrame(
            {
                "row_index": row_indices,
                "Drug ID": drug_ids,
                "Protein ID": protein_ids,
                "Label": labels_np.astype(int),
                "Logit": logits_np,
                "Probability": probabilities_np,
                "Prediction": (probabilities_np >= threshold).astype(int),
            }
        ).sort_values("row_index", kind="stable")
    else:
        prediction_frame = pd.DataFrame()
    return EpochResult(metrics=metrics, predictions=prediction_frame)


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class ExperimentTrainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        config: dict[str, Any],
        input_dims: dict[str, dict[str, int]],
        output_dir: Path,
        device: torch.device,
        logger: logging.Logger,
        retain_checkpoints: bool = True,
        collect_train_predictions: bool = False,
        write_artifacts: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.input_dims = input_dims
        self.output_dir = output_dir
        self.device = device
        self.logger = logger
        self.retain_checkpoints = retain_checkpoints
        self.collect_train_predictions = collect_train_predictions
        self.write_artifacts = write_artifacts
        if not write_artifacts and retain_checkpoints:
            raise ValueError("不写训练文件时必须使用内存最佳状态（retain_checkpoints=False）。")
        self.best_payload: dict[str, Any] | None = None
        self.training_config = config["training"]
        self.threshold = float(self.training_config["decision_threshold"])
        self.precision = str(self.training_config["mixed_precision"])
        if (
            self.precision == "bf16"
            and device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "当前 GPU 不支持 BF16，但配置 training.mixed_precision=bf16。"
                "请改为 fp16 后重新启动。"
            )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=float(self.training_config["learning_rate"]),
            weight_decay=float(self.training_config["weight_decay"]),
        )
        self.base_learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.warmup_config = self.training_config.get("warmup", {})
        self.warmup_enabled = bool(self.warmup_config.get("enabled", False))
        if self.warmup_enabled:
            warmup_epochs = int(self.warmup_config.get("epochs", 0))
            start_factor = float(self.warmup_config.get("start_factor", 0.1))
            if warmup_epochs <= 0 or not 0.0 < start_factor <= 1.0:
                raise ValueError("warmup 要求 epochs > 0 且 start_factor 位于 (0, 1]。")
            for group, base_lr in zip(self.optimizer.param_groups, self.base_learning_rates):
                group["lr"] = base_lr * start_factor
        scheduler_config = self.training_config["scheduler"]
        if scheduler_config["name"] != "reduce_on_plateau":
            raise ValueError("当前主实验只支持 reduce_on_plateau 动态学习率策略。")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=str(self.training_config["monitor_mode"]),
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            threshold=float(scheduler_config["threshold"]),
            threshold_mode=str(scheduler_config.get("threshold_mode", "abs")),
            min_lr=float(scheduler_config["min_lr"]),
        )
        scaler_enabled = self.precision == "fp16" and device.type == "cuda"
        self.scaler = _make_grad_scaler(enabled=scaler_enabled)
        self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if write_artifacts else _NullWriter()
        self.checkpoint_dir = output_dir / "checkpoints"
        self.prediction_dir = output_dir / "predictions"
        if self.retain_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if write_artifacts:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_name = str(self.training_config["monitor"])
        self.monitor_mode = str(self.training_config["monitor_mode"])
        self.early_config = self.training_config["early_stopping"]
        self.best_value = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.epochs_without_improvement = 0
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []
        ema_config = self.training_config.get("ema", {})
        self.ema = (
            ExponentialMovingAverage(self.model, float(ema_config.get("decay", 0.999)))
            if ema_config.get("enabled", False)
            else None
        )

    def _monitor_value(self, validation_metrics: dict[str, Any]) -> float:
        metric_name = self.monitor_name.removeprefix("val_")
        if metric_name not in validation_metrics:
            raise KeyError(f"监控指标不存在：{self.monitor_name}")
        return float(validation_metrics[metric_name])

    def _is_improved(self, value: float) -> bool:
        delta = float(self.early_config["min_delta"])
        if self.monitor_mode == "max":
            return value > self.best_value + delta
        return value < self.best_value - delta

    def _step_scheduler(self, monitored: float, epoch: int) -> None:
        if self.warmup_enabled and epoch <= int(self.warmup_config["epochs"]):
            progress = epoch / int(self.warmup_config["epochs"])
            start_factor = float(self.warmup_config["start_factor"])
            factor = start_factor + (1.0 - start_factor) * progress
            for group, base_lr in zip(self.optimizer.param_groups, self.base_learning_rates):
                group["lr"] = base_lr * factor
        else:
            self.scheduler.step(monitored)

    @contextmanager
    def _ema_evaluation(self):
        if self.ema is None:
            yield
        else:
            with self.ema.average_parameters(self.model):
                yield

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        train_generator = getattr(self.loaders["train"], "generator", None)
        return {
            "format_version": 1,
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "scaler_state": self.scaler.state_dict(),
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "config": self.config,
            "input_dims": self.input_dims,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_loader_generator": (
                    train_generator.get_state() if train_generator is not None else None
                ),
            },
        }

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        if not self.retain_checkpoints:
            if path.name == "best_checkpoint.pt":
                # Clone both live and EMA parameters: CPU tensors must not alias the
                # training model. Keep the same best-epoch selection as the main run.
                self.best_payload = {
                    "epoch": epoch,
                    "best_value": self.best_value,
                    "model_state": {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    },
                    "ema_state": (
                        {
                            "decay": self.ema.decay,
                            "shadow": {
                                name: value.detach().cpu().clone()
                                for name, value in self.ema.shadow.items()
                            },
                        }
                        if self.ema is not None else None
                    ),
                }
            return
        atomic_torch_save(self._checkpoint_payload(epoch), path)

    def resume(self, checkpoint_path: str | Path) -> None:
        if not self.retain_checkpoints:
            raise ValueError("内存 checkpoint 模式不能断点续训；请从头重跑未完成的 seed。")
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch_load(checkpoint_path, map_location="cpu")
        saved_architecture = payload.get("config", {}).get("model", {}).get("architecture")
        expected_architecture = self.config.get("model", {}).get("architecture")
        if saved_architecture != expected_architecture:
            raise ValueError("checkpoint 模型版本与当前实验不同，禁止跨版本续训。")
        target_best = self.checkpoint_dir / "best_checkpoint.pt"
        source_best = checkpoint_path.parent / "best_checkpoint.pt"
        if not target_best.is_file():
            if source_best.is_file():
                atomic_torch_save(torch_load(source_best, map_location="cpu"), target_best)
            else:
                raise FileNotFoundError(
                    "断点续训还需要与 last_checkpoint.pt 同目录的 best_checkpoint.pt。"
                    "请复制完整 checkpoints 目录后重试。"
                )
        # RNG 与 DataLoader Generator 状态必须保留为 CPU ByteTensor；模型和优化器
        # 状态随后分别由 load_state_dict / _move_optimizer_state 放到正确设备。
        checkpoint_dataset = (
            payload.get("config", {}).get("experiment", {}).get("dataset")
        )
        current_dataset = self.config.get("experiment", {}).get("dataset")
        if checkpoint_dataset != current_dataset:
            raise ValueError(
                f"checkpoint 数据集为 {checkpoint_dataset!r}，当前数据集为 "
                f"{current_dataset!r}，禁止跨数据集续训。"
            )
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state(self.optimizer, self.device)
        self.scheduler.load_state_dict(payload["scheduler_state"])
        ema_state = payload.get("ema_state")
        if self.ema is not None:
            if ema_state is None:
                raise ValueError("当前实验启用了 EMA，但断点中没有 EMA 状态。")
            self.ema.load_state_dict(ema_state, self.model)
        scaler_state = payload.get("scaler_state")
        if scaler_state:
            self.scaler.load_state_dict(scaler_state)
        self.best_value = float(payload["best_value"])
        self.epochs_without_improvement = int(payload["epochs_without_improvement"])
        self.history = list(payload.get("history", []))
        self.start_epoch = int(payload["epoch"]) + 1
        rng = payload.get("rng_state", {})
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
            train_generator = getattr(self.loaders["train"], "generator", None)
            if train_generator is not None and rng.get("train_loader_generator") is not None:
                train_generator.set_state(rng["train_loader_generator"].cpu())
        self.logger.info("已从 %s 恢复，将从 epoch %d 继续。", checkpoint_path, self.start_epoch)

    def fit(self, resume_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
        if resume_path is not None:
            self.resume(resume_path)
        max_epochs = int(self.training_config["max_epochs"])
        patience = int(self.early_config["patience"])
        best_checkpoint = self.checkpoint_dir / "best_checkpoint.pt"
        last_checkpoint = self.checkpoint_dir / "last_checkpoint.pt"

        try:
            for epoch in range(self.start_epoch, max_epochs + 1):
                epoch_start = time.perf_counter()
                train_result = run_epoch(
                    self.model,
                    self.loaders["train"],
                    self.criterion,
                    self.device,
                    self.threshold,
                    self.precision,
                    split="train",
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    gradient_accumulation_steps=int(
                        self.training_config["gradient_accumulation_steps"]
                    ),
                    gradient_clip_norm=float(self.training_config["gradient_clip_norm"]),
                    collect_predictions=False,
                    ema=self.ema,
                )
                with self._ema_evaluation():
                    val_result = run_epoch(
                        self.model,
                        self.loaders["val"],
                        self.criterion,
                        self.device,
                        self.threshold,
                        self.precision,
                        split="val",
                        collect_predictions=False,
                    )
                monitored = self._monitor_value(val_result.metrics)
                # 这是刚完成的 epoch 实际使用的学习率；scheduler.step 计算下一
                # 个 epoch 的学习率，二者分开记录，避免日志语义错位。
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                self._step_scheduler(monitored, epoch)
                next_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                improved = self._is_improved(monitored)
                if improved:
                    self.best_value = monitored
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                elapsed = time.perf_counter() - epoch_start
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "next_learning_rate": next_learning_rate,
                    "elapsed_seconds": elapsed,
                    "improved": improved,
                }
                row.update({f"train_{key}": value for key, value in train_result.metrics.items()})
                row.update({f"val_{key}": value for key, value in val_result.metrics.items()})
                self.history.append(row)
                if self.write_artifacts:
                    pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)

                lr_message = f"lr={learning_rate:.8g}"
                if next_learning_rate != learning_rate:
                    lr_message += f" -> next_lr={next_learning_rate:.8g}"
                self.logger.info("Epoch %03d/%03d | %s | %.1fs", epoch, max_epochs, lr_message, elapsed)
                self.logger.info(format_metrics("train", train_result.metrics))
                self.logger.info(format_metrics("val", val_result.metrics))
                self.logger.info(
                    "监控 %s=%.6f | best=%.6f | 未提升轮数=%d/%d%s",
                    self.monitor_name,
                    monitored,
                    self.best_value,
                    self.epochs_without_improvement,
                    patience,
                    (" | 保存最佳权重" if self.retain_checkpoints else " | 记录内存最佳状态")
                    if improved else "",
                )
                for key, value in train_result.metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val_result.metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"val/{key}", value, epoch)
                self.writer.add_scalar("train/learning_rate", learning_rate, epoch)
                self.writer.flush()

                if improved:
                    self._save_checkpoint(best_checkpoint, epoch)
                if bool(self.training_config.get("save_last_checkpoint", True)):
                    self._save_checkpoint(last_checkpoint, epoch)
                if self.epochs_without_improvement >= patience:
                    self.logger.info("触发早停：验证集监控指标连续 %d 个 epoch 未提升。", patience)
                    break
        finally:
            self.writer.close()

        if self.retain_checkpoints:
            if not best_checkpoint.is_file():
                raise RuntimeError("训练结束但未生成最佳权重文件。")
            best_payload = torch_load(best_checkpoint, map_location="cpu")
        else:
            if self.best_payload is None:
                raise RuntimeError("训练结束但未记录最佳模型状态。")
            best_payload = self.best_payload
        self.model.load_state_dict(best_payload["model_state"])
        if self.ema is not None:
            ema_state = best_payload.get("ema_state")
            if ema_state is None:
                raise ValueError("最佳 checkpoint 缺少 EMA 状态。")
            self.ema.load_state_dict(ema_state, self.model)
            self.ema.copy_to(self.model)
        self.logger.info(
            "已加载最佳权重：epoch=%d，%s=%.6f",
            int(best_payload["epoch"]),
            self.monitor_name,
            float(best_payload["best_value"]),
        )
        results = self.final_evaluation()
        if self.retain_checkpoints:
            self._export_best_weights(best_payload)
        return results

    def _export_best_weights(self, best_payload: dict[str, Any]) -> None:
        atomic_torch_save(
            {
                "model_state": {
                    name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()
                },
                "config": self.config,
                "input_dims": self.input_dims,
                "best_epoch": int(best_payload["epoch"]),
                "best_value": float(best_payload["best_value"]),
                "decision_threshold": float(self.threshold),
            },
            self.checkpoint_dir / "best_model_weights.pt",
        )

    def final_evaluation(self) -> dict[str, dict[str, Any]]:
        self.logger.info("开始使用最佳权重评估训练集、验证集和测试集。")
        results: dict[str, dict[str, Any]] = {}
        for split, loader_name in (("train", "train_eval"), ("val", "val"), ("test", "test")):
            result = run_epoch(
                self.model,
                self.loaders[loader_name],
                self.criterion,
                self.device,
                self.threshold,
                self.precision,
                split=split,
                collect_predictions=self.write_artifacts and (self.collect_train_predictions or split in {"val", "test"}),
            )
            results[split] = result.metrics
            self.logger.info(format_metrics(split, result.metrics))
            if self.write_artifacts and (self.collect_train_predictions or split in {"val", "test"}):
                prediction_path = self.prediction_dir / f"{split}_predictions.csv"
                result.predictions.to_csv(prediction_path, index=False)
                self.logger.info("已保存 %s 预测结果：%s", split, prediction_path)

        if self.write_artifacts:
            write_json(results, self.output_dir / "metrics_final.json")
            rows = [{"split": split, **metrics} for split, metrics in results.items()]
            pd.DataFrame(rows).to_csv(self.output_dir / "metrics_final.csv", index=False)
        return results
