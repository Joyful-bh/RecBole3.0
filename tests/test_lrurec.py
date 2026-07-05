from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from recbole3.dataset import ML100KRetrievalConfig, ML100KRetrievalDataset, SplitConfig
from recbole3.evaluation import EvalConfig
from recbole3.model import get_model_spec
from recbole3.model.lrurec import LRURecConfig, LRURecModel, LRURecModelDataset
from recbole3.model.lrurec.data import LRU_INPUT_IDS, LRU_LABEL_IDS
from recbole3.model.lrurec.trainer import LRURecTrainer, LRURecTrainerConfig


def _write_dataset(path: Path, *, num_items: int = 7) -> Path:
    artifact = {
        "train": {1: [1, 2, 3, 4, 5], 2: [2, 3]},
        "val": {1: [6], 2: [4]},
        "test": {1: [7], 2: [5]},
        "meta": {item_id: f"Item {item_id}" for item_id in range(1, num_items + 1)},
        "umap": {"u1": 1, "u2": 2},
        "smap": {item_id: item_id for item_id in range(1, num_items + 1)},
    }
    dataset_path = path / "dataset.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return dataset_path


def _prepared_data(tmp_path: Path, dataset_path: Path):
    task_data = ML100KRetrievalDataset(
        ML100KRetrievalConfig(
            mapping_source=str(dataset_path),
            processed_dir=str(tmp_path / "processed"),
            split=SplitConfig(
                strategy="leave_one_out",
                order="chronological",
                per_user=True,
                valid_holdout_num=1,
                test_holdout_num=1,
            ),
        )
    ).prepare(eval_config=EvalConfig(protocol="full"))
    return LRURecModelDataset.from_task_dataset(
        task_data,
        model_config=LRURecConfig(history_max_length=3, sliding_window_size=1.0),
    )


def test_lrurec_dataset_reproduces_original_windows(tmp_path: Path) -> None:
    prepared = _prepared_data(tmp_path, _write_dataset(tmp_path))
    train_frame = prepared.get_train_dataset().frame
    assert train_frame[[LRU_INPUT_IDS, LRU_LABEL_IDS]].to_dict("records") == [
        {LRU_INPUT_IDS: (2, 3, 4), LRU_LABEL_IDS: (3, 4, 5)},
        {LRU_INPUT_IDS: (2,), LRU_LABEL_IDS: (2, 3)},
    ]
    assert prepared.get_eval_dataset("valid").frame[LRU_INPUT_IDS].tolist() == [(1, 2, 3, 4, 5), (2, 3)]
    assert prepared.get_eval_dataset("test").frame[LRU_INPUT_IDS].tolist() == [(1, 2, 3, 4, 5, 6), (2, 3, 4)]


def test_lrurec_collators_use_left_padding_and_original_ids(tmp_path: Path) -> None:
    prepared = _prepared_data(tmp_path, _write_dataset(tmp_path))
    model = LRURecModel(LRURecConfig(history_max_length=3))
    train_batch = model.build_train_collator(prepared)(prepared.get_train_dataset().frame)
    assert train_batch[LRU_INPUT_IDS].tolist() == [[2, 3, 4], [0, 0, 2]]
    assert train_batch[LRU_LABEL_IDS].tolist() == [[3, 4, 5], [0, 2, 3]]
    eval_batch = model.build_eval_collator(prepared)(prepared.get_eval_dataset("test").frame)
    assert eval_batch[LRU_INPUT_IDS].tolist() == [[4, 5, 6], [2, 3, 4]]


def test_lrurec_model_shapes_and_padding_class(tmp_path: Path) -> None:
    prepared = _prepared_data(tmp_path, _write_dataset(tmp_path))
    model = LRURecModel(LRURecConfig(history_max_length=3, hidden_size=8, num_blocks=1, dropout=0.0, attention_dropout=0.0))
    batch = model.build_train_collator(prepared)(prepared.get_train_dataset().frame)
    outputs = model.forward(batch)
    assert outputs["logits"].shape == (2, 3, 8)
    assert torch.isfinite(model.compute_loss(batch, outputs))
    predictions = model.predict(
        {LRU_INPUT_IDS: batch[LRU_INPUT_IDS]},
        k=3,
        exclude_item_ids=torch.tensor([[2, 3], [1, 2]]),
        exclude_mask=torch.ones((2, 2), dtype=torch.bool),
    )
    assert predictions.shape == (2, 3)


def test_lrurec_is_registered() -> None:
    spec = get_model_spec("lrurec")
    assert spec.model_cls is LRURecModel


def test_lrurec_batch_average_metrics_match_source_equal_batch_weighting() -> None:
    labels = np.arange(18, dtype=np.int64)
    predictions = np.tile(np.arange(100, 150, dtype=np.int64), (18, 1))
    predictions[:16, 0] = labels[:16]

    overall, batch_average = LRURecTrainer._compute_metric_views(
        predictions,
        labels,
        batch_size=16,
    )

    assert overall["recall@10"] == 16 / 18
    assert batch_average["recall@10"] == 0.5


class _ExportOnlyLRURecTrainer(LRURecTrainer):
    """Avoid a training loop while exercising the real final-export path."""

    def _fit_original(self, model, prepared_data, output_dir):
        model.ensure_initialized(prepared_data)
        checkpoint = output_dir / "best_model.pt"
        torch.save(model.state_dict(), checkpoint)
        return {
            "train_history": [],
            "valid_history": [],
            "stopped_early": False,
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_step": 0,
            "best_metric": {"name": "recall@10", "value": 0.0},
            "checkpoint_paths": {"best": str(checkpoint), "last": str(checkpoint)},
        }


def _run_export_with_exclude_history(tmp_path: Path, *, exclude_history: bool) -> tuple[dict, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    prepared = _prepared_data(tmp_path, _write_dataset(tmp_path, num_items=60))
    model = LRURecModel(LRURecConfig(history_max_length=3, hidden_size=8, num_blocks=1, dropout=0.0, attention_dropout=0.0))

    def fixed_scores(batch):
        # Items 3, 4, and 5 are in user 1's valid history after H=3 truncation.
        score = torch.arange(61, dtype=torch.float32, device=batch[LRU_INPUT_IDS].device)
        score[0] = -1e9
        score[3:6] = torch.tensor([101.0, 102.0, 103.0], device=score.device)
        return score.unsqueeze(0).expand(batch[LRU_INPUT_IDS].shape[0], -1).clone()

    model.last_item_scores = fixed_scores  # type: ignore[method-assign]
    config = LRURecTrainerConfig(batch_size=2, candidate_topk=3, save_full_score_matrix=False)
    config.eval.exclude_history = exclude_history
    output_dir = tmp_path / f"exclude_{exclude_history}"
    _ExportOnlyLRURecTrainer(config).run(model, prepared, output_dir=output_dir)

    with (output_dir / "retrieved.pkl").open("rb") as handle:
        retrieved = pickle.load(handle)
    metrics = json.loads((output_dir / "stage1_metrics.json").read_text(encoding="utf-8"))
    return retrieved, metrics


def test_lrurec_final_export_honors_exclude_history_and_records_effective_settings(tmp_path: Path) -> None:
    unfiltered, unfiltered_metrics = _run_export_with_exclude_history(tmp_path / "unfiltered", exclude_history=False)
    filtered, filtered_metrics = _run_export_with_exclude_history(tmp_path / "filtered", exclude_history=True)

    assert 5 in unfiltered["val_candidate_ids"][0]
    assert not ({3, 4, 5} & set(filtered["val_candidate_ids"][0]))
    assert "val_probs" not in unfiltered and "test_probs" not in unfiltered

    for expected, retrieved, metrics in (
        (False, unfiltered, unfiltered_metrics),
        (True, filtered, filtered_metrics),
    ):
        assert retrieved["export_config"] == {
            "history_max_length": 3,
            "exclude_history": expected,
            "candidate_topk": 3,
            "save_full_score_matrix": False,
            "candidate_export_exclude_history": expected,
        }
        assert metrics["export_config"]["history_max_length"] == 3
        assert metrics["export_config"]["candidate_topk"] == 3
        assert metrics["export_config"]["save_full_score_matrix"] is False
        assert metrics["export_config"]["early_stop_eval_exclude_history"] is expected
        assert metrics["export_config"]["final_eval_exclude_history"] is expected
        assert metrics["export_config"]["candidate_export_exclude_history"] is expected
        assert metrics["valid"]["exclude_history"] is expected
        assert metrics["test"]["exclude_history"] is expected
