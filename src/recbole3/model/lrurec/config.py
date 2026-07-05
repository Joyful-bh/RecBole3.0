from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class LRURecConfig(SequentialModelConfig):
    name: str = field(default="lrurec", metadata={"help": "LlamaRec stage-1 LRURec retriever."})
    history_max_length: int = field(default=200, metadata={"help": "Original ML-100K LRU sequence length."})
    hidden_size: int = field(default=64, metadata={"help": "Token and LRU hidden dimension."})
    num_blocks: int = field(default=2, metadata={"help": "Number of LRU plus feed-forward blocks."})
    dropout: float = field(default=0.4, metadata={"help": "Embedding and feed-forward dropout."})
    attention_dropout: float = field(default=0.4, metadata={"help": "LRU projection dropout."})
    sliding_window_size: float = field(default=1.0, metadata={"help": "Training-window stride as a fraction of max length."})
    initializer_std: float = field(default=0.02, metadata={"help": "Truncated-normal initializer standard deviation."})
    r_min: float = field(default=0.8, metadata={"help": "Minimum initial LRU eigenvalue radius."})
    r_max: float = field(default=0.99, metadata={"help": "Maximum initial LRU eigenvalue radius."})


__all__ = ["LRURecConfig"]
