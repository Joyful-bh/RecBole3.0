from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from recbole3.dataset import BaseTaskDataset, FrameDataset, ITEM_ID, USER_ID



def build_llamarec_artifact_from_task_data(
    task_data: BaseTaskDataset,
    *,
    item_text_field: str,
) -> dict[str, Any]:
    """Build a LlamaRec-compatible dataset.pkl artifact from prepared RecBole data."""

    num_users = int(task_data.get_num_users())
    num_items = int(task_data.get_num_items())
    user_inverse = _invert_dense_map(_require_id_map(task_data, "_user_id_map"), expected_size=num_users, name="user")
    item_inverse = _invert_dense_map(_require_id_map(task_data, "_item_id_map"), expected_size=num_items, name="item")

    artifact: dict[str, Any] = {
        "train": {user_id + 1: [] for user_id in range(num_users)},
        "val": {user_id + 1: [] for user_id in range(num_users)},
        "test": {user_id + 1: [] for user_id in range(num_users)},
        "meta": _build_meta(task_data, item_text_field=item_text_field),
        "umap": {user_inverse[user_id]: user_id + 1 for user_id in range(num_users)},
        "smap": {item_inverse[item_id]: item_id + 1 for item_id in range(num_items)},
    }
    _append_split_items(artifact["train"], _frame(task_data.get_train_dataset()))
    _append_split_items(artifact["val"], _frame(task_data.get_eval_dataset("valid")))
    _append_split_items(artifact["test"], _frame(task_data.get_eval_dataset("test")))
    return artifact


def load_or_build_llamarec_artifact(
    task_data: BaseTaskDataset,
    *,
    item_text_field: str,
    refresh_cache: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Load or create one cached LlamaRec-compatible dataset.pkl for a prepared dataset."""

    cache_dir = _artifact_cache_dir(task_data, item_text_field=item_text_field)
    artifact_path = cache_dir / "dataset.pkl"
    manifest_path = cache_dir / "manifest.json"
    if artifact_path.exists() and not bool(refresh_cache):
        with artifact_path.open("rb") as handle:
            return pickle.load(handle), artifact_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_llamarec_artifact_from_task_data(task_data, item_text_field=item_text_field)
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    manifest_path.write_text(
        json.dumps(_artifact_signature_payload(task_data, item_text_field=item_text_field), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return artifact, artifact_path


def _append_split_items(target: dict[int, list[int]], frame: Any) -> None:
    if frame.empty:
        return
    for record in frame.to_dict("records"):
        user_id = int(record[USER_ID]) + 1
        item_id = int(record[ITEM_ID]) + 1
        target.setdefault(user_id, []).append(item_id)


def _build_meta(task_data: BaseTaskDataset, *, item_text_field: str) -> dict[int, str]:
    item_table = task_data.get_item_table()
    if item_text_field not in item_table.columns:
        raise ValueError(f"LlamaRec artifact requires item text field '{item_text_field}' in the item table.")
    meta = {item_id + 1: f"item {item_id + 1}" for item_id in range(int(task_data.get_num_items()))}
    for record in item_table.to_dict("records"):
        item_id = int(record[ITEM_ID])
        value = record.get(item_text_field)
        text = "" if value is None else str(value).strip()
        meta[item_id + 1] = text or f"item {item_id + 1}"
    return meta


def _artifact_cache_dir(task_data: BaseTaskDataset, *, item_text_field: str) -> Path:
    parser = getattr(task_data, "_parser", None)
    fallback_dir = Path(str(getattr(task_data.config, "processed_dir", "data/processed"))) / str(task_data.config.name)
    try:
        data_dir = Path(parser.data_dir) if parser is not None else fallback_dir
    except NotImplementedError:
        data_dir = fallback_dir
    signature = hashlib.sha1(
        json.dumps(_artifact_signature_payload(task_data, item_text_field=item_text_field), sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return data_dir / "llamarec" / signature


def _artifact_signature_payload(task_data: BaseTaskDataset, *, item_text_field: str) -> dict[str, Any]:
    config = task_data.config
    config_payload = asdict(config) if is_dataclass(config) else dict(getattr(config, "__dict__", {}))
    return {
        "implementation_version": "llamarec-artifact-v1",
        "dataset_config": _json_safe(config_payload),
        "item_text_field": str(item_text_field),
        "num_users": int(task_data.get_num_users()),
        "num_items": int(task_data.get_num_items()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _require_id_map(task_data: BaseTaskDataset, name: str) -> Mapping[Any, int]:
    mapping = getattr(task_data, name, None)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"LlamaRec artifact generation requires prepared dataset {name}.")
    return mapping


def _invert_dense_map(mapping: Mapping[Any, int], *, expected_size: int, name: str) -> dict[int, Any]:
    inverse: dict[int, Any] = {}
    for raw_id, dense_id in mapping.items():
        dense_id = int(dense_id)
        if dense_id in inverse:
            raise ValueError(f"Duplicate RecBole {name}_id mapping for dense id {dense_id}.")
        inverse[dense_id] = raw_id
    expected = set(range(int(expected_size)))
    actual = set(inverse)
    if actual != expected:
        missing = sorted(expected.difference(actual))[:5]
        extra = sorted(actual.difference(expected))[:5]
        raise ValueError(f"RecBole {name}_id map is not dense 0-based; missing={missing}, extra={extra}.")
    return inverse


def _frame(dataset: Any) -> Any:
    if not isinstance(dataset, FrameDataset):
        raise TypeError(f"LlamaRec artifact generation requires FrameDataset splits, got {type(dataset).__name__}.")
    return dataset.frame.copy()


__all__ = [
    "build_llamarec_artifact_from_task_data",
    "load_or_build_llamarec_artifact",
]
