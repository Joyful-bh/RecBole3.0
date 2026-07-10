from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Literal

import pandas as pd
import torch

from recbole3.dataset import BaseTaskDataset, CANDIDATE_ITEM_IDS, FrameDataset, ITEM_ID
from recbole3.dataset.ml100k import ORIGINAL_ITEM_ID, ORIGINAL_USER_ID, RAW_ITEM_ID


TARGET_IN_CANDIDATES = "target_in_candidates"


def import_llamarec_candidates(
    task_data: BaseTaskDataset,
    *,
    retrieved_path: str | Path,
    mapping_source: str | Path,
    topk: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import LlamaRec-compatible candidates without changing their id semantics."""

    mapping = _load_pickle(Path(mapping_source), description="LlamaRec dataset.pkl")
    retrieved = _load_pickle(Path(retrieved_path), description="LlamaRec retrieved.pkl")
    _validate_mapping(mapping)
    num_users = len(mapping["umap"])
    num_items = len(mapping["smap"])
    if int(topk) <= 0 or int(topk) > num_items:
        raise ValueError(f"candidate topk must be in [1, {num_items}], got {topk}.")
    inverse_smap = {int(internal_id): raw_id for raw_id, internal_id in mapping["smap"].items()}
    prepared_items = task_data.get_item_table().sort_values(ORIGINAL_ITEM_ID).reset_index(drop=True)
    if prepared_items[ORIGINAL_ITEM_ID].astype(int).tolist() != list(range(1, num_items + 1)):
        raise ValueError("Prepared item table does not preserve dense original_item_id order from dataset.pkl.")
    prepared_raw_item_ids = prepared_items[RAW_ITEM_ID].tolist()
    frames = []
    for split in ("val", "test"):
        eval_split: Literal["valid", "test"] = "valid" if split == "val" else "test"
        eval_dataset = task_data.get_eval_dataset(eval_split)
        if not isinstance(eval_dataset, FrameDataset):
            raise TypeError(f"LlamaRec candidate import requires FrameDataset, got {type(eval_dataset).__name__}.")
        eval_frame = eval_dataset.frame.copy().reset_index(drop=True)
        probs = retrieved.get(f"{split}_probs")
        imported_candidate_ids = retrieved.get(f"{split}_candidate_ids")
        labels = retrieved.get(f"{split}_labels")
        if probs is not None:
            _validate_retrieved_matrix(
                probs,
                labels,
                split=split,
                num_users=num_users,
                num_items=num_items,
            )
        else:
            _validate_compact_candidates(
                imported_candidate_ids,
                labels,
                split=split,
                num_users=num_users,
                num_items=num_items,
                topk=topk,
            )
        if len(eval_frame) != num_users:
            raise ValueError(
                f"Prepared {eval_split} split has {len(eval_frame)} rows, expected one row for each of {num_users} users."
            )

        candidate_rows: list[tuple[int, ...]] = []
        hit_rows: list[bool] = []
        for row_index, record in eval_frame.iterrows():
            original_user_id = int(record[ORIGINAL_USER_ID])
            expected_user_id = row_index + 1
            if original_user_id != expected_user_id:
                raise ValueError(
                    f"Prepared {eval_split} row {row_index} has original_user_id={original_user_id}, "
                    f"expected {expected_user_id}."
                )
            target_original_item_id = int(record[ORIGINAL_ITEM_ID])
            if int(labels[row_index]) != target_original_item_id:
                raise ValueError(
                    f"{split}_labels row {row_index} is {labels[row_index]}, "
                    f"but prepared target original_item_id is {target_original_item_id}."
                )
            if probs is not None:
                score_row = torch.as_tensor(probs[row_index])
                original_candidate_ids = torch.topk(score_row, k=int(topk)).indices.tolist()
            else:
                original_candidate_ids = [
                    int(item_id) for item_id in imported_candidate_ids[row_index][: int(topk)]
                ]
            if any(item_id < 1 or item_id > num_items for item_id in original_candidate_ids):
                raise ValueError(f"{split} row {row_index} contains padding or out-of-range top-{topk} candidates.")
            for original_item_id in original_candidate_ids:
                expected_raw_item_id = inverse_smap[int(original_item_id)]
                prepared_raw_item_id = prepared_raw_item_ids[int(original_item_id) - 1]
                if prepared_raw_item_id != expected_raw_item_id:
                    raise ValueError(
                        f"Item mapping mismatch for original_item_id={original_item_id}: "
                        f"dataset.pkl raw id={expected_raw_item_id!r}, prepared raw id={prepared_raw_item_id!r}."
                    )
            candidate_rows.append(tuple(int(item_id) - 1 for item_id in original_candidate_ids))
            hit_rows.append(target_original_item_id in original_candidate_ids)

        eval_frame[CANDIDATE_ITEM_IDS] = candidate_rows
        eval_frame[TARGET_IN_CANDIDATES] = hit_rows
        frames.append(eval_frame)
    return frames[0], frames[1]


def _load_pickle(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found at {path}.")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a dictionary, got {type(value).__name__}.")
    return value


def _validate_mapping(mapping: dict[str, Any]) -> None:
    required = {"train", "val", "test", "umap", "smap"}
    missing = required.difference(mapping)
    if missing:
        raise ValueError(f"LlamaRec dataset.pkl is missing keys: {sorted(missing)}")
    for name in ("umap", "smap"):
        values = sorted(int(value) for value in mapping[name].values())
        if values != list(range(1, len(values) + 1)):
            raise ValueError(f"LlamaRec {name} is not a dense 1-based mapping.")


def _validate_retrieved_matrix(
    probs: Any,
    labels: Any,
    *,
    split: str,
    num_users: int,
    num_items: int,
) -> None:
    if probs is None or labels is None:
        raise ValueError(f"retrieved.pkl is missing {split}_probs or {split}_labels.")
    if len(probs) != num_users or len(labels) != num_users:
        raise ValueError(
            f"retrieved.pkl {split} rows must equal num_users={num_users}; "
            f"got probs={len(probs)}, labels={len(labels)}."
        )
    expected_width = num_items + 1
    bad_widths = [len(row) for row in probs if len(row) != expected_width]
    if bad_widths:
        raise ValueError(
            f"retrieved.pkl {split}_probs must have width {expected_width} including padding column 0."
        )
    if any(float(row[0]) > -1e8 for row in probs):
        raise ValueError(f"retrieved.pkl {split}_probs padding column 0 is not masked.")


def _validate_compact_candidates(
    candidate_ids: Any,
    labels: Any,
    *,
    split: str,
    num_users: int,
    num_items: int,
    topk: int,
) -> None:
    if candidate_ids is None or labels is None:
        raise ValueError(
            f"candidate artifact must contain either {split}_probs or "
            f"{split}_candidate_ids together with {split}_labels."
        )
    if len(candidate_ids) != num_users or len(labels) != num_users:
        raise ValueError(
            f"compact {split} rows must equal num_users={num_users}; "
            f"got candidates={len(candidate_ids)}, labels={len(labels)}."
        )
    for row_index, row in enumerate(candidate_ids):
        if len(row) < int(topk):
            raise ValueError(f"compact {split} row {row_index} has fewer than topk={topk} candidates.")
        selected = [int(item_id) for item_id in row[: int(topk)]]
        if len(set(selected)) != len(selected):
            raise ValueError(f"compact {split} row {row_index} contains duplicate candidates.")
        if any(item_id < 1 or item_id > num_items for item_id in selected):
            raise ValueError(f"compact {split} row {row_index} contains an out-of-range original item ID.")


__all__ = [
    "TARGET_IN_CANDIDATES",
    "import_llamarec_candidates",
]
