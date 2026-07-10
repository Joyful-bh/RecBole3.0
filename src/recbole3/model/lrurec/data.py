from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from recbole3.config import instantiate_dataclass
from recbole3.dataset import FrameDataset, ITEM_ID
from recbole3.dataset.ml100k import ORIGINAL_ITEM_ID, ORIGINAL_USER_ID
from recbole3.model.base import BaseCollator, ModelConfig, ModelDatasets
from recbole3.model.lrurec.config import LRURecConfig
from recbole3.model.sequential import BaseSequentialModelDataset


LRU_INPUT_IDS = "lru_input_ids"
LRU_LABEL_IDS = "lru_label_ids"


class LRURecModelDataset(BaseSequentialModelDataset):
    """Rebuild the exact windowed LRURec inputs from the original dataset.pkl."""

    def _build_model_datasets(self, *, model_config: ModelConfig) -> ModelDatasets:
        if not isinstance(model_config, LRURecConfig):
            model_config = instantiate_dataclass(LRURecConfig, model_config)
        mapping_source = Path(str(getattr(self.config, "mapping_source", "")))
        if not mapping_source.is_file():
            raise FileNotFoundError(f"LRURec mapping_source not found at {mapping_source}.")
        with mapping_source.open("rb") as handle:
            artifact = pickle.load(handle)
        required = {"train", "val", "test", "umap", "smap"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"LRURec dataset.pkl is missing keys: {sorted(missing)}")

        max_length = int(model_config.history_max_length)
        sliding_step = int(float(model_config.sliding_window_size) * max_length)
        if max_length <= 0 or sliding_step <= 0:
            raise ValueError("LRURec history_max_length and sliding step must be positive.")

        train_rows: list[dict[str, Any]] = []
        for original_user_id in sorted(int(user_id) for user_id in artifact["train"]):
            sequence = [int(item_id) for item_id in artifact["train"][original_user_id]]
            windows = [sequence]
            if len(sequence) >= max_length + sliding_step:
                windows = [
                    sequence[start : start + max_length]
                    for start in range(len(sequence) - max_length, -1, -sliding_step)
                ]
            for window in windows:
                labels = window[-max_length:]
                tokens = window[:-1][-max_length:]
                train_rows.append(
                    {
                        ORIGINAL_USER_ID: original_user_id,
                        LRU_INPUT_IDS: tuple(tokens),
                        LRU_LABEL_IDS: tuple(labels),
                    }
                )

        valid_rows = self._build_eval_rows(artifact, split="valid")
        test_rows = self._build_eval_rows(artifact, split="test")
        return ModelDatasets(
            train_dataset=FrameDataset(pd.DataFrame(train_rows)),
            valid_dataset=FrameDataset(valid_rows),
            test_dataset=FrameDataset(test_rows),
        )

    @staticmethod
    def _build_eval_rows(artifact: dict[str, Any], *, split: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for original_user_id in sorted(int(user_id) for user_id in artifact["train"]):
            sequence = list(artifact["train"][original_user_id])
            answers = artifact["val"][original_user_id]
            if split == "test":
                sequence += list(artifact["val"][original_user_id])
                answers = artifact["test"][original_user_id]
            if not answers:
                continue
            target_original_item_id = int(answers[0])
            rows.append(
                {
                    ORIGINAL_USER_ID: original_user_id,
                    ORIGINAL_ITEM_ID: target_original_item_id,
                    ITEM_ID: target_original_item_id - 1,
                    LRU_INPUT_IDS: tuple(int(item_id) for item_id in sequence),
                }
            )
        return pd.DataFrame(rows)


class LRURecTrainCollator(BaseCollator):
    config: LRURecConfig

    def __call__(self, records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = records.to_dict("records") if isinstance(records, pd.DataFrame) else list(records)
        max_length = int(self.config.history_max_length)
        input_rows = [_left_pad(row[LRU_INPUT_IDS], max_length) for row in rows]
        label_rows = [_left_pad(row[LRU_LABEL_IDS], max_length) for row in rows]
        return {
            LRU_INPUT_IDS: torch.tensor(input_rows, dtype=torch.long),
            LRU_LABEL_IDS: torch.tensor(label_rows, dtype=torch.long),
        }


class LRURecEvalCollator(BaseCollator):
    config: LRURecConfig

    def __call__(self, records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        rows = records.to_dict("records") if isinstance(records, pd.DataFrame) else list(records)
        max_length = int(self.config.history_max_length)
        return {
            LRU_INPUT_IDS: torch.tensor(
                [_left_pad(row[LRU_INPUT_IDS], max_length) for row in rows],
                dtype=torch.long,
            )
        }


def _left_pad(values: Sequence[int], width: int) -> list[int]:
    values = [int(value) for value in values][-int(width) :]
    return [0] * (int(width) - len(values)) + values


__all__ = [
    "LRU_INPUT_IDS",
    "LRU_LABEL_IDS",
    "LRURecEvalCollator",
    "LRURecModelDataset",
    "LRURecTrainCollator",
]
