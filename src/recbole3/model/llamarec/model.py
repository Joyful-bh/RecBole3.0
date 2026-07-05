from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from recbole3.model.llamarec.config import LlamaRecConfig


class LlamaRecModel:
    """Prompt and verbalizer helpers used by the QLoRA reranker."""

    def __init__(self, config: LlamaRecConfig):
        self.config = config

    def build_prompt(
        self,
        history_titles: Sequence[str],
        candidate_titles: Sequence[str],
        *,
        tokenizer: Any | None = None,
    ) -> str:
        history = " \n ".join(
            f"({index + 1}) {self._truncate_title(title, tokenizer)}"
            for index, title in enumerate(history_titles[-int(self.config.history_max_length) :])
        )
        candidates = " \n ".join(
            f"({chr(ord('A') + index)}) {self._truncate_title(title, tokenizer)}"
            for index, title in enumerate(candidate_titles)
        )
        prompt_input = self.config.input_template.format(history, candidates)
        return (
            f"### Instruction:\n{self.config.system_template}\n\n"
            f"### Input:\n{prompt_input}\n\n### Response: "
        )

    def _truncate_title(self, title: str, tokenizer: Any | None) -> str:
        if tokenizer is None:
            return str(title)
        tokens = tokenizer.tokenize(str(title))[: int(self.config.max_title_tokens)]
        return tokenizer.convert_tokens_to_string(tokens)


def rank_verbalizer_scores(
    letter_scores: torch.Tensor,
    *,
    k: int,
    stable_candidate_ties: bool,
) -> torch.Tensor:
    """Rank letters by score, preserving retrieved order when scores are equal."""

    if letter_scores.ndim != 2:
        raise ValueError(f"Expected rank-2 verbalizer scores, got shape {tuple(letter_scores.shape)}.")
    candidate_count = int(letter_scores.shape[1])
    if int(k) <= 0 or int(k) > candidate_count:
        raise ValueError(f"k must be in [1, {candidate_count}], got {k}.")
    if stable_candidate_ties:
        return torch.argsort(letter_scores, dim=1, descending=True, stable=True)[:, : int(k)]
    return torch.topk(letter_scores, k=int(k), dim=1).indices


__all__ = ["LlamaRecModel", "rank_verbalizer_scores"]