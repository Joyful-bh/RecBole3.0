from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from recbole3.dataset.cache import DatasetCache
from recbole3.dataset.config import DatasetConfig
from recbole3.dataset.parser import BaseDatasetParser, ParsedData
from recbole3.dataset.utils import ITEM_ID, LABEL, TIMESTAMP, USER_ID


RAW_USER_ID = "raw_user_id"
RAW_ITEM_ID = "raw_item_id"
ORIGINAL_USER_ID = "original_user_id"
ORIGINAL_ITEM_ID = "original_item_id"


@dataclass(slots=True)
class ML100KBaseConfig(DatasetConfig):
    name: str = field(default="", metadata={"help": "MovieLens latest-small retrieval dataset name."})
    mapping_source: str = field(
        default="/home/wangxiaolei/xujiale/LlamaRec/data/preprocessed/ml-100k_min_rating0-min_uc5-min_sc5/dataset.pkl",
        metadata={"help": "Required original LlamaRec dataset.pkl used to preserve umap/smap exactly."},
    )
    processed_dir: str = field(default="data/processed", metadata={"help": "Processed cache root."})
    refresh_cache: bool = field(default=False, metadata={"help": "Whether to rebuild parser-managed caches."})


class ML100KBaseParser(BaseDatasetParser):
    """Build RecBole3 records from the original LlamaRec preprocessed artifact."""

    config_cls = ML100KBaseConfig
    config: ML100KBaseConfig

    def parse(self) -> ParsedData:
        mapping_path = Path(self.config.mapping_source)
        if not mapping_path.is_file():
            raise FileNotFoundError(
                "LlamaRec-compatible ML-100K requires the original dataset.pkl mapping artifact. "
                f"mapping_source not found at {mapping_path}; refusing to generate a replacement mapping."
            )
        cache = self._parsed_cache()
        if not self.config.refresh_cache and cache.parsed_exists():
            return cache.read_parsed()
        artifact = self._load_mapping_artifact(mapping_path)
        parsed = self._build_parsed_data(artifact)
        cache.write_parsed(parsed)
        return parsed

    @property
    def data_dir(self) -> Path:
        return self._parsed_root_dir()

    def _parsed_root_dir(self) -> Path:
        return Path(self.config.processed_dir) / self.config.name / "llamarec_mapping"

    def _parsed_cache(self) -> DatasetCache:
        return DatasetCache(self._parsed_root_dir())

    @staticmethod
    def _load_mapping_artifact(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        required = {"train", "val", "test", "meta", "umap", "smap"}
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"LlamaRec dataset.pkl is missing keys: {sorted(missing)}")
        ML100KBaseParser._validate_dense_mapping(artifact["umap"], name="umap")
        ML100KBaseParser._validate_dense_mapping(artifact["smap"], name="smap")
        return artifact

    @staticmethod
    def _validate_dense_mapping(mapping: dict[Any, Any], *, name: str) -> None:
        values = sorted(int(value) for value in mapping.values())
        expected = list(range(1, len(mapping) + 1))
        if values != expected:
            raise ValueError(f"LlamaRec {name} must contain dense 1-based ids from 1 to {len(mapping)}.")

    def _build_parsed_data(self, artifact: dict[str, Any]) -> ParsedData:
        umap = artifact["umap"]
        smap = artifact["smap"]
        inverse_umap = {int(internal_id): raw_id for raw_id, internal_id in umap.items()}
        inverse_smap = {int(internal_id): raw_id for raw_id, internal_id in smap.items()}
        user_table = pd.DataFrame(
            [
                {
                    USER_ID: inverse_umap[original_user_id],
                    RAW_USER_ID: inverse_umap[original_user_id],
                    ORIGINAL_USER_ID: original_user_id,
                }
                for original_user_id in range(1, len(umap) + 1)
            ]
        )
        metadata = artifact["meta"]
        item_table = pd.DataFrame(
            [
                {
                    ITEM_ID: inverse_smap[original_item_id],
                    RAW_ITEM_ID: inverse_smap[original_item_id],
                    ORIGINAL_ITEM_ID: original_item_id,
                    "title": str(metadata.get(original_item_id, "")),
                    "metadata_text": str(metadata.get(original_item_id, "")),
                }
                for original_item_id in range(1, len(smap) + 1)
            ]
        )

        rows: list[dict[str, Any]] = []
        for original_user_id in range(1, len(umap) + 1):
            raw_user_id = inverse_umap[original_user_id]
            sequence = (
                list(artifact["train"][original_user_id])
                + list(artifact["val"][original_user_id])
                + list(artifact["test"][original_user_id])
            )
            for position, original_item_id in enumerate(sequence, start=1):
                original_item_id = int(original_item_id)
                rows.append(
                    {
                        USER_ID: raw_user_id,
                        ITEM_ID: inverse_smap[original_item_id],
                        TIMESTAMP: position,
                        LABEL: None,
                        RAW_USER_ID: raw_user_id,
                        RAW_ITEM_ID: inverse_smap[original_item_id],
                        ORIGINAL_USER_ID: original_user_id,
                        ORIGINAL_ITEM_ID: original_item_id,
                    }
                )
        return ParsedData(
            interactions=pd.DataFrame(rows),
            user_table=user_table,
            item_table=item_table,
        )


__all__ = [
    "ML100KBaseConfig",
    "ML100KBaseParser",
    "ORIGINAL_ITEM_ID",
    "ORIGINAL_USER_ID",
    "RAW_ITEM_ID",
    "RAW_USER_ID",
]
