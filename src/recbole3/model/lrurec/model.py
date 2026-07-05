from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from recbole3.model.base import BaseCollator, BaseRetrievalModel
from recbole3.model.lrurec.config import LRURecConfig
from recbole3.model.lrurec.data import LRU_INPUT_IDS, LRU_LABEL_IDS, LRURecEvalCollator, LRURecTrainCollator


class LRURecModel(BaseRetrievalModel):
    """Original LlamaRec LRURec with a RecBole3 retrieval boundary."""

    def __init__(self, config: LRURecConfig):
        super().__init__(config)
        self.config = config
        self.embedding: LRUEmbedding | None = None
        self.model: LRUModel | None = None
        self._num_items: int | None = None

    def ensure_initialized(self, prepared_data: Any) -> None:
        self._ensure_initialized(int(prepared_data.get_num_items()))

    def build_train_collator(self, prepared_data: Any) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return LRURecTrainCollator(self.config, prepared_data)

    def build_eval_collator(self, prepared_data: Any) -> BaseCollator:
        self._ensure_initialized(int(prepared_data.get_num_items()))
        return LRURecEvalCollator(self.config, prepared_data)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"logits": self.score_sequences(batch[LRU_INPUT_IDS])}

    def compute_loss(self, batch: Mapping[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = outputs["logits"]
        labels = batch[LRU_LABEL_IDS].to(device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=0)

    def predict(
        self,
        model_inputs: Mapping[str, torch.Tensor],
        *,
        k: int,
        candidate_item_ids: torch.Tensor | None = None,
        exclude_item_ids: torch.Tensor | None = None,
        exclude_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.last_item_scores(model_inputs)
        real_item_scores = scores[:, 1:]
        if candidate_item_ids is not None:
            candidates = candidate_item_ids.to(device=real_item_scores.device, dtype=torch.long)
            candidate_scores = real_item_scores.gather(1, candidates)
            positions = torch.topk(candidate_scores, k=int(k), dim=1).indices
            return candidates.gather(1, positions)
        if exclude_item_ids is not None and exclude_mask is not None and exclude_item_ids.numel() > 0:
            history_mask = torch.zeros_like(real_item_scores, dtype=torch.bool)
            history_mask.scatter_(
                1,
                exclude_item_ids.to(device=real_item_scores.device, dtype=torch.long),
                exclude_mask.to(device=real_item_scores.device, dtype=torch.bool),
            )
            real_item_scores = real_item_scores.masked_fill(history_mask, -1e9)
        return torch.topk(real_item_scores, k=int(k), dim=1).indices

    def last_item_scores(self, model_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.score_sequences(model_inputs[LRU_INPUT_IDS])[:, -1, :]

    def score_sequences(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding, model = self._lru_modules()
        input_ids = input_ids.to(device=embedding.token.weight.device, dtype=torch.long)
        hidden, mask = embedding(input_ids)
        return model(hidden, embedding.token.weight, mask)

    def _ensure_initialized(self, num_items: int) -> None:
        if self.embedding is not None:
            if self._num_items != int(num_items):
                raise ValueError(f"LRURec was initialized for {self._num_items} items, received {num_items}.")
            return
        self._num_items = int(num_items)
        self.embedding = LRUEmbedding(self.config, num_items=int(num_items))
        self.model = LRUModel(self.config, num_items=int(num_items))
        self._truncated_normal_init()

    def _lru_modules(self) -> tuple["LRUEmbedding", "LRUModel"]:
        if self.embedding is None or self.model is None:
            raise RuntimeError("LRURecModel must be initialized with prepared_data first.")
        return self.embedding, self.model

    def _truncated_normal_init(self) -> None:
        mean = 0.0
        std = float(self.config.initializer_std)
        lower, upper = -2 * std, 2 * std
        with torch.no_grad():
            lower_cdf = (1.0 + math.erf(((lower - mean) / std) / math.sqrt(2.0))) / 2.0
            upper_cdf = (1.0 + math.erf(((upper - mean) / std) / math.sqrt(2.0))) / 2.0
            for name, parameter in self.named_parameters():
                if "layer_norm" in name or "params_log" in name:
                    continue
                components = (parameter.real, parameter.imag) if torch.is_complex(parameter) else (parameter,)
                for component in components:
                    component.uniform_(2 * lower_cdf - 1, 2 * upper_cdf - 1)
                    component.erfinv_()
                    component.mul_(std * math.sqrt(2.0))
                    component.add_(mean)


class LRUEmbedding(nn.Module):
    def __init__(self, config: LRURecConfig, *, num_items: int):
        super().__init__()
        self.token = nn.Embedding(int(num_items) + 1, int(config.hidden_size))
        self.layer_norm = nn.LayerNorm(int(config.hidden_size))
        self.embed_dropout = nn.Dropout(float(config.dropout))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = input_ids > 0
        hidden = self.token(input_ids)
        return self.layer_norm(self.embed_dropout(hidden)), mask


class LRUModel(nn.Module):
    def __init__(self, config: LRURecConfig, *, num_items: int):
        super().__init__()
        self.lru_blocks = nn.ModuleList([LRUBlock(config) for _ in range(int(config.num_blocks))])
        self.bias = nn.Parameter(torch.zeros(int(num_items) + 1))

    def forward(self, hidden: torch.Tensor, embedding_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence_length = int(hidden.shape[1])
        padded_length = 2 ** int(np.ceil(np.log2(sequence_length)))
        hidden = F.pad(hidden, (0, 0, padded_length - sequence_length, 0, 0, 0))
        padded_mask = F.pad(mask, (padded_length - sequence_length, 0, 0, 0))
        for block in self.lru_blocks:
            hidden = block(hidden, padded_mask)
        hidden = hidden[:, -sequence_length:]
        return torch.matmul(hidden, embedding_weight.permute(1, 0)) + self.bias


class LRUBlock(nn.Module):
    def __init__(self, config: LRURecConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.lru_layer = LRULayer(
            d_model=hidden_size,
            dropout=float(config.attention_dropout),
            r_min=float(config.r_min),
            r_max=float(config.r_max),
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size,
            d_ff=hidden_size * 4,
            dropout=float(config.dropout),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.lru_layer(hidden, mask))


class LRULayer(nn.Module):
    def __init__(self, d_model: int, dropout: float, r_min: float, r_max: float, use_bias: bool = True):
        super().__init__()
        self.embed_size = int(d_model)
        self.hidden_size = 2 * int(d_model)
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diagonal = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diagonal) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        self.out_vector = nn.Identity()
        self.dropout = nn.Dropout(p=float(dropout))
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        recurrent = self.in_proj(hidden.to(torch.cfloat)) * gamma
        batch_size, sequence_length, hidden_size = recurrent.shape
        for index in range(int(np.ceil(np.log2(sequence_length)))):
            recurrent, lamb = self._parallel_step(
                index + 1,
                recurrent,
                lamb,
                mask,
                batch_size,
                sequence_length,
                hidden_size,
            )
        output = self.dropout(self.out_proj(recurrent).real) + self.out_vector(hidden)
        return self.layer_norm(output)

    @staticmethod
    def _parallel_step(
        index: int,
        hidden: torch.Tensor,
        lamb: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = 2**index
        hidden = hidden.reshape(batch_size * sequence_length // width, width, hidden_size)
        step_mask = mask.reshape(batch_size * sequence_length // width, width)
        first, second = hidden[:, : width // 2], hidden[:, width // 2 :]
        if index > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        second = second + lamb * first[:, -1:] * step_mask[:, width // 2 - 1 : width // 2].unsqueeze(-1)
        return torch.cat([first, second], dim=1), lamb


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        transformed = self.dropout(self.activation(self.w_1(hidden)))
        return self.layer_norm(self.dropout(self.w_2(transformed)) + hidden)


__all__ = ["LRURecModel"]
