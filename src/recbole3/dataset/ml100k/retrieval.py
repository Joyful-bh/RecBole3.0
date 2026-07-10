from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.dataset.base import BaseTaskDataset

from .base import ML100KBaseConfig, ML100KBaseParser


@dataclass(slots=True)
class ML100KRetrievalConfig(ML100KBaseConfig):
    name: str = field(default="ml100k_retrieval", metadata={"help": "LlamaRec-compatible ML-100K dataset name."})


class ML100KRetrievalParser(ML100KBaseParser):
    config_cls = ML100KRetrievalConfig
    config: ML100KRetrievalConfig


class ML100KRetrievalDataset(BaseTaskDataset):
    config_cls = ML100KRetrievalConfig
    parser_cls = ML100KRetrievalParser


__all__ = [
    "ML100KRetrievalConfig",
    "ML100KRetrievalDataset",
    "ML100KRetrievalParser",
]
