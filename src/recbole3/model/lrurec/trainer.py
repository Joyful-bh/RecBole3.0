from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import pickle
import random
import time
from typing import Any, Literal

import numpy as np
import torch

from recbole3.dataset import FrameDataset
from recbole3.evaluation.config import EvalConfig
from recbole3.evaluation.metric import MRRMetric, MetricSpec, NDCGMetric, RecallMetric, RetrievalEvalData
from recbole3.model.base import BaseModel
from recbole3.model.lrurec.data import LRU_INPUT_IDS
from recbole3.model.lrurec.model import LRURecModel
from recbole3.trainer import Trainer
from recbole3.trainer_config import CheckpointConfig, EarlyStoppingConfig, OptimizerConfig, TrainerConfig


@dataclass(slots=True)
class LRURecTrainerConfig(TrainerConfig):
    batch_size: int = field(default=16, metadata={"help": "Original ML-100K train batch size."})
    shuffle: bool = field(default=True, metadata={"help": "Shuffle original sequence windows."})
    pin_memory: bool = field(default=True, metadata={"help": "Pin host batches for CUDA transfer."})
    max_epochs: int = field(default=500, metadata={"help": "Original maximum epoch count."})
    validation_interval_steps: int = field(default=500, metadata={"help": "Original validation interval."})
    seed: int = field(default=42, metadata={"help": "Original LlamaRec random seed."})
    max_grad_norm: float = field(default=5.0, metadata={"help": "Original gradient clipping norm."})
    candidate_filename: str = field(default="retrieved.pkl", metadata={"help": "Exported LlamaRec-compatible stage-1 artifact."})
    candidate_topk: int = field(default=20, metadata={"help": "Number of original item IDs stored per user."})
    save_full_score_matrix: bool = field(
        default=True,
        metadata={"help": "Also store the 610 x 3651 score matrices for strict legacy compatibility."},
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            name="AdamW",
            kwargs={"lr": 0.001, "weight_decay": 0.01, "eps": 1e-9},
        )
    )
    monitor: str | None = field(default="recall@10")
    early_stopping: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig(enabled=True, patience=20, min_delta=0.0)
    )
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(save_best=True, save_last=True)
    )
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            exclude_history=True,
            metrics=(
                MetricSpec(name="recall", ks=(1, 5, 10, 20, 50)),
                MetricSpec(name="ndcg", ks=(1, 5, 10, 20, 50)),
                MetricSpec(name="mrr", ks=(1, 5, 10, 20, 50)),
            ),
            neg_sampling_num=0,
        ),
        metadata={"help": "Original full-item stage-1 metrics."},
    )


class LRURecTrainer(Trainer):
    config_cls = LRURecTrainerConfig

    def run(
        self,
        model: BaseModel,
        prepared_data: Any,
        *,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if not isinstance(model, LRURecModel):
            raise TypeError(f"LRURecTrainer requires LRURecModel, got {type(model).__name__}.")
        output_path = Path(output_dir or "outputs/lrurec")
        output_path.mkdir(parents=True, exist_ok=True)
        self._set_seed(int(self.config.seed))
        self._setup_logger(model, prepared_data, output_path)
        started = time.perf_counter()
        try:
            export_settings = self._export_settings(model)
            configured_exclude_history = bool(export_settings["exclude_history"])
            print(
                "[lrurec:config] "
                f"history_max_length={export_settings['history_max_length']} "
                f"exclude_history={configured_exclude_history} "
                f"candidate_topk={export_settings['candidate_topk']} "
                f"save_full_score_matrix={export_settings['save_full_score_matrix']}"
            )
            fit_result = self._fit_original(model, prepared_data, output_path)
            best_path = Path(fit_result["checkpoint_paths"]["best"])
            model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
            device = self._device()
            model.to(device)
            test_result, test_scores, test_labels = self._evaluate_scores(
                model,
                prepared_data,
                split="test",
                exclude_history=configured_exclude_history,
            )
            valid_result, valid_scores, valid_labels = self._evaluate_scores(
                model,
                prepared_data,
                split="valid",
                exclude_history=configured_exclude_history,
            )
            self._assert_export_evaluation_settings(
                valid_result,
                test_result,
                configured_exclude_history=configured_exclude_history,
            )
            candidate_path = output_path / self.config.candidate_filename
            candidate_payload = {
                "format": "recbole3_lrurec_topk_v1",
                "topk": int(self.config.candidate_topk),
                "export_config": {
                    **export_settings,
                    "candidate_export_exclude_history": configured_exclude_history,
                },
                "val_candidate_ids": torch.topk(
                    torch.from_numpy(valid_scores),
                    k=int(self.config.candidate_topk),
                    dim=1,
                ).indices.tolist(),
                "val_labels": valid_labels.tolist(),
                "val_metrics": self._original_metric_names(valid_result["metrics"]),
                "test_candidate_ids": torch.topk(
                    torch.from_numpy(test_scores),
                    k=int(self.config.candidate_topk),
                    dim=1,
                ).indices.tolist(),
                "test_labels": test_labels.tolist(),
                "test_metrics": self._original_metric_names(test_result["metrics"]),
            }
            if self.config.save_full_score_matrix:
                candidate_payload["val_probs"] = valid_scores.tolist()
                candidate_payload["test_probs"] = test_scores.tolist()
            with candidate_path.open("wb") as handle:
                pickle.dump(candidate_payload, handle)
            (output_path / "stage1_metrics.json").write_text(
                json.dumps(
                    {
                        "export_config": {
                            **export_settings,
                            "early_stop_eval_exclude_history": configured_exclude_history,
                            "final_eval_exclude_history": configured_exclude_history,
                            "candidate_export_exclude_history": configured_exclude_history,
                        },
                        "fit": fit_result,
                        "valid": valid_result,
                        "test": test_result,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_test(test_result)
                logger.log_summary(
                    stopped_early=fit_result["stopped_early"],
                    total_epochs=fit_result["epochs_completed"],
                    best_epoch=fit_result["best_epoch"],
                    total_time=time.perf_counter() - started,
                )
            return {
                "fit": fit_result,
                "test": test_result,
                "valid_candidates": valid_result,
                "candidate_path": str(candidate_path),
            }
        finally:
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.close()

    def _fit_original(self, model: LRURecModel, prepared_data: Any, output_dir: Path) -> dict[str, Any]:
        device = self._device()
        model.to(device)
        collator = model.build_train_collator(prepared_data)
        train_loader = self.build_dataloader(
            prepared_data.get_train_dataset(),
            collator,
            shuffle=bool(self.config.shuffle),
        )
        optimizer = self._build_original_optimizer(model)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best_model.pt"
        last_path = checkpoint_dir / "last_model.pt"

        best_value = float("-inf")
        best_epoch = 0
        best_step = 0
        bad_count = 0
        global_step = 0
        stopped_early = False
        validation_history: list[dict[str, Any]] = []
        train_history: list[dict[str, Any]] = []
        configured_exclude_history = bool(self.config.eval.exclude_history)

        initial_result, _, _ = self._evaluate_scores(
            model,
            prepared_data,
            split="valid",
            exclude_history=configured_exclude_history,
        )
        best_value = float(initial_result["batch_average_metrics"]["recall@10"])
        torch.save(model.state_dict(), best_path)
        validation_history.append({"epoch": 0, "step": 0, **initial_result})

        for epoch in range(1, int(self.config.max_epochs) + 1):
            epoch_start = time.perf_counter()
            model.train()
            losses: list[float] = []
            progress = self._create_train_progress_bar(
                train_loader,
                epoch=epoch,
                max_epochs=int(self.config.max_epochs),
                disable=False,
            )
            for batch in progress:
                batch = {name: value.to(device) for name, value in batch.items()}
                optimizer.zero_grad()
                outputs = model.forward(batch)
                loss = model.compute_loss(batch, outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.max_grad_norm))
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().cpu()))

                if global_step % int(self.config.validation_interval_steps) == 0:
                    valid_result, _, _ = self._evaluate_scores(
                        model,
                        prepared_data,
                        split="valid",
                        exclude_history=configured_exclude_history,
                    )
                    value = float(valid_result["batch_average_metrics"]["recall@10"])
                    improved = value > best_value + float(self.config.early_stopping.min_delta)
                    if improved:
                        best_value = value
                        best_epoch = epoch
                        best_step = global_step
                        bad_count = 0
                        torch.save(model.state_dict(), best_path)
                    else:
                        bad_count += 1
                    validation_history.append({"epoch": epoch, "step": global_step, **valid_result})
                    print(
                        f"[lrurec:valid] epoch={epoch} step={global_step} "
                        f"batch_recall@10={value:.9f} "
                        f"overall_recall@10={valid_result['metrics']['recall@10']:.9f} "
                        f"best={best_value:.9f} bad={bad_count}"
                    )
                    if (logger := getattr(self, "_logger", None)) is not None:
                        logger.log_validation(epoch=epoch, metrics=valid_result["batch_average_metrics"])
                    model.train()
                    if self.config.early_stopping.enabled and bad_count >= int(self.config.early_stopping.patience):
                        stopped_early = True
                        break
            if hasattr(progress, "close"):
                progress.close()
            epoch_loss = float(np.mean(losses)) if losses else None
            train_history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "num_batches": len(losses),
                    "elapsed_seconds": time.perf_counter() - epoch_start,
                    "global_step": global_step,
                }
            )
            torch.save(model.state_dict(), last_path)
            if (logger := getattr(self, "_logger", None)) is not None:
                logger.log_epoch(
                    epoch=epoch,
                    max_epochs=int(self.config.max_epochs),
                    loss=epoch_loss,
                    num_batches=len(losses),
                    elapsed_seconds=time.perf_counter() - epoch_start,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    global_step=float(global_step),
                )
            if stopped_early:
                break

        return {
            "train_history": train_history,
            "valid_history": validation_history,
            "stopped_early": stopped_early,
            "epochs_completed": len(train_history),
            "best_epoch": best_epoch,
            "best_step": best_step,
            "best_metric": {"name": "recall@10", "value": best_value},
            "checkpoint_paths": {"best": str(best_path), "last": str(last_path)},
        }

    def _evaluate_scores(
        self,
        model: LRURecModel,
        prepared_data: Any,
        *,
        split: Literal["valid", "test"],
        exclude_history: bool,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        dataset = prepared_data.get_eval_dataset(split)
        if not isinstance(dataset, FrameDataset):
            raise TypeError(f"LRURec evaluation requires FrameDataset, got {type(dataset).__name__}.")
        collator = model.build_eval_collator(prepared_data)
        loader = self.build_dataloader(dataset, collator, shuffle=False)
        device = self._device()
        model.eval()
        score_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = {name: value.to(device) for name, value in batch.items()}
                scores = model.last_item_scores(batch)
                sequences = batch[LRU_INPUT_IDS]
                if exclude_history:
                    rows = torch.arange(scores.shape[0], device=device).unsqueeze(1).expand_as(sequences)
                    scores[rows, sequences] = -1e9
                    scores[:, 0] = -1e9
                score_rows.append(scores.detach().cpu())
        scores = torch.cat(score_rows, dim=0)
        labels = dataset.frame["original_item_id"].astype(int).to_numpy()
        max_k = max(k for spec in self.config.eval.metrics for k in spec.ks)
        predictions = torch.topk(scores, k=max_k, dim=1).indices.numpy()
        metrics, batch_average_metrics = self._compute_metric_views(
            predictions,
            labels,
            batch_size=int(self.config.batch_size),
        )
        return {
            "split": split,
            "metrics": metrics,
            "batch_average_metrics": batch_average_metrics,
            "exclude_history": exclude_history,
            "num_users": int(len(labels)),
            "num_batches": int(np.ceil(len(labels) / int(self.config.batch_size))),
        }, scores.numpy(), labels

    def _export_settings(self, model: LRURecModel) -> dict[str, int | bool]:
        return {
            "history_max_length": int(model.config.history_max_length),
            "exclude_history": bool(self.config.eval.exclude_history),
            "candidate_topk": int(self.config.candidate_topk),
            "save_full_score_matrix": bool(self.config.save_full_score_matrix),
        }

    @staticmethod
    def _assert_export_evaluation_settings(
        valid_result: dict[str, Any],
        test_result: dict[str, Any],
        *,
        configured_exclude_history: bool,
    ) -> None:
        for split, result in (("valid", valid_result), ("test", test_result)):
            if bool(result["exclude_history"]) != configured_exclude_history:
                raise AssertionError(
                    "LRURec final "
                    f"{split} evaluation exclude_history={result['exclude_history']} does not match "
                    f"trainer.eval.exclude_history={configured_exclude_history}."
                )

    @classmethod
    def _compute_metric_views(
        cls,
        predictions: np.ndarray,
        labels: np.ndarray,
        *,
        batch_size: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if batch_size <= 0:
            raise ValueError("LRURec metric batch_size must be positive.")
        overall = cls._compute_metrics(predictions, labels)
        batch_metrics = [
            cls._compute_metrics(predictions[start : start + batch_size], labels[start : start + batch_size])
            for start in range(0, len(labels), batch_size)
        ]
        batch_average = {
            key: float(np.mean([metrics[key] for metrics in batch_metrics]))
            for key in overall
        }
        return overall, batch_average

    @staticmethod
    def _compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        eval_data = RetrievalEvalData(
            pred_item_ids=predictions,
            target_item_ids=labels.reshape(-1, 1),
            target_mask=np.ones((len(labels), 1), dtype=bool),
        )
        metrics: dict[str, float] = {}
        ks = (1, 5, 10, 20, 50)
        for metric in (RecallMetric(ks), NDCGMetric(ks), MRRMetric(ks)):
            metrics.update(metric.compute(eval_data))
        return metrics

    def _build_original_optimizer(self, model: LRURecModel) -> torch.optim.Optimizer:
        kwargs = dict(self.config.optimizer.kwargs)
        weight_decay = float(kwargs.pop("weight_decay", 0.0))
        no_decay = ("bias", "layer_norm")
        named_parameters = list(model.named_parameters())
        groups = [
            {
                "params": [p for name, p in named_parameters if not any(key in name for key in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for name, p in named_parameters if any(key in name for key in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(groups, **kwargs)

    @staticmethod
    def _device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None and accelerator.is_available():
            return torch.device(str(accelerator.current_accelerator().type))
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _original_metric_names(metrics: dict[str, float]) -> dict[str, float]:
        return {
            f"{name.upper() if name != 'ndcg' else 'NDCG'}@{k}": float(value)
            for key, value in metrics.items()
            for name, k in [key.split("@", 1)]
        }


__all__ = ["LRURecTrainer", "LRURecTrainerConfig"]
