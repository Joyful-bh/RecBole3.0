from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from recbole3.dataset import ITEM_ID, USER_ID, ML100KRetrievalConfig, ML100KRetrievalDataset, SplitConfig
from recbole3.dataset.ml100k import ORIGINAL_ITEM_ID, ORIGINAL_USER_ID, RAW_ITEM_ID, RAW_USER_ID
from recbole3.evaluation import EvalConfig
from recbole3.model.llamarec.candidates import TARGET_IN_CANDIDATES, import_llamarec_candidates
from recbole3.model.llamarec.config import LlamaRecConfig
from recbole3.model.llamarec.data import build_llamarec_artifact_from_task_data, load_or_build_llamarec_artifact
from recbole3.model.llamarec.model import LlamaRecModel, rank_verbalizer_scores
from recbole3.model.llamarec.trainer import (
    build_external_train_dataloader,
    build_ranker_feature,
    collate_ranker_features,
)
from tests.test_helpers import StubDataset, StubDatasetConfig


def _write_artifacts(root: Path) -> tuple[Path, Path]:
    dataset_path = root / "dataset.pkl"
    retrieved_path = root / "retrieved.pkl"
    dataset = {
        "train": {1: [1], 2: [2]},
        "val": {1: [2], 2: [3]},
        "test": {1: [3], 2: [4]},
        "meta": {1: "One (1995)", 2: "Two (1996)", 3: "Three (1997)", 4: "Four (1998)"},
        "umap": {"raw-u2": 1, "raw-u1": 2},
        "smap": {122882: 1, 1: 2, 99: 3, 2: 4},
    }
    retrieved = {
        "val_probs": [
            [-1e9, 0.1, 0.9, 0.2, 0.0],
            [-1e9, 0.1, 0.2, 0.9, 0.0],
        ],
        "val_labels": [2, 3],
        "val_metrics": {},
        "test_probs": [
            [-1e9, 0.8, 0.7, 0.9, 0.6],
            [-1e9, 0.9, 0.8, 0.7, 0.1],
        ],
        "test_labels": [3, 4],
        "test_metrics": {},
    }
    with dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle)
    with retrieved_path.open("wb") as handle:
        pickle.dump(retrieved, handle)
    return dataset_path, retrieved_path


def _dataset_config(root: Path, mapping_source: Path) -> ML100KRetrievalConfig:
    return ML100KRetrievalConfig(
        mapping_source=str(mapping_source),
        processed_dir=str(root / "processed"),
        split=SplitConfig(
            strategy="leave_one_out",
            order="chronological",
            per_user=True,
            valid_holdout_num=1,
            test_holdout_num=1,
        ),
    )


def test_ml100k_parser_preserves_original_mapping_order(tmp_path: Path) -> None:
    dataset_path, _ = _write_artifacts(tmp_path)
    prepared = ML100KRetrievalDataset(_dataset_config(tmp_path, dataset_path)).prepare(
        eval_config=EvalConfig(protocol="full")
    )

    users = prepared.get_user_table()
    items = prepared.get_item_table()
    assert users[[USER_ID, RAW_USER_ID, ORIGINAL_USER_ID]].to_dict("records") == [
        {USER_ID: 0, RAW_USER_ID: "raw-u2", ORIGINAL_USER_ID: 1},
        {USER_ID: 1, RAW_USER_ID: "raw-u1", ORIGINAL_USER_ID: 2},
    ]
    assert items[[ITEM_ID, RAW_ITEM_ID, ORIGINAL_ITEM_ID]].to_dict("records") == [
        {ITEM_ID: 0, RAW_ITEM_ID: 122882, ORIGINAL_ITEM_ID: 1},
        {ITEM_ID: 1, RAW_ITEM_ID: 1, ORIGINAL_ITEM_ID: 2},
        {ITEM_ID: 2, RAW_ITEM_ID: 99, ORIGINAL_ITEM_ID: 3},
        {ITEM_ID: 3, RAW_ITEM_ID: 2, ORIGINAL_ITEM_ID: 4},
    ]
    assert prepared.get_eval_dataset("test").frame[ITEM_ID].tolist() == [2, 3]


def test_ml100k_parser_refuses_missing_mapping_source(tmp_path: Path) -> None:
    dataset = ML100KRetrievalDataset(_dataset_config(tmp_path, tmp_path / "missing.pkl"))
    with pytest.raises(FileNotFoundError, match="refusing to generate"):
        dataset.prepare(eval_config=EvalConfig(protocol="full"))


def test_llamarec_artifact_builder_converts_recbole_frames_to_qlora_schema(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig(processed_dir=str(tmp_path / "processed"))).prepare(
        eval_config=EvalConfig(protocol="full")
    )

    artifact = build_llamarec_artifact_from_task_data(prepared, item_text_field="metadata_text")

    assert artifact["train"] == {1: [1, 2], 2: [5, 6]}
    assert artifact["val"] == {1: [3], 2: [7]}
    assert artifact["test"] == {1: [4], 2: [8]}
    assert artifact["umap"] == {0: 1, 1: 2}
    assert artifact["smap"] == {item_id: item_id + 1 for item_id in range(8)}
    assert artifact["meta"][1] == "Alpha Quest"
    assert artifact["meta"][8] == "Ivory Path"


def test_llamarec_artifact_cache_round_trips_generated_dataset(tmp_path: Path) -> None:
    prepared = StubDataset(StubDatasetConfig(processed_dir=str(tmp_path / "processed"))).prepare(
        eval_config=EvalConfig(protocol="full")
    )

    first, path = load_or_build_llamarec_artifact(prepared, item_text_field="metadata_text", refresh_cache=True)
    second, second_path = load_or_build_llamarec_artifact(prepared, item_text_field="metadata_text")

    assert path == second_path
    assert path.name == "dataset.pkl"
    assert first == second
    assert (path.parent / "manifest.json").exists()


def test_retrieved_import_validates_mapping_and_keeps_misses(tmp_path: Path) -> None:
    dataset_path, retrieved_path = _write_artifacts(tmp_path)
    prepared = ML100KRetrievalDataset(_dataset_config(tmp_path, dataset_path)).prepare(
        eval_config=EvalConfig(protocol="full")
    )

    _, test_frame = import_llamarec_candidates(
        prepared,
        retrieved_path=retrieved_path,
        mapping_source=dataset_path,
        topk=2,
    )

    assert test_frame["candidate_item_ids"].tolist() == [(2, 0), (0, 1)]
    assert test_frame[TARGET_IN_CANDIDATES].tolist() == [True, False]
    assert len(test_frame) == 2


def test_compact_retrieved_import_preserves_candidate_order(tmp_path: Path) -> None:
    dataset_path, retrieved_path = _write_artifacts(tmp_path)
    with retrieved_path.open("rb") as handle:
        retrieved = pickle.load(handle)
    retrieved.pop("val_probs")
    retrieved.pop("test_probs")
    retrieved["val_candidate_ids"] = [[2, 3], [3, 2]]
    retrieved["test_candidate_ids"] = [[3, 1], [1, 2]]
    with retrieved_path.open("wb") as handle:
        pickle.dump(retrieved, handle)
    prepared = ML100KRetrievalDataset(_dataset_config(tmp_path, dataset_path)).prepare(
        eval_config=EvalConfig(protocol="full")
    )

    _, test_frame = import_llamarec_candidates(
        prepared,
        retrieved_path=retrieved_path,
        mapping_source=dataset_path,
        topk=2,
    )

    assert test_frame["candidate_item_ids"].tolist() == [(2, 0), (0, 1)]
    assert test_frame[TARGET_IN_CANDIDATES].tolist() == [True, False]

class _ToyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and "A" <= text <= "T":
            return [32 + ord(text) - ord("A")]
        return [ord(char) for char in text]

    def __call__(
        self,
        text,
        truncation=False,
        padding=False,
        add_special_tokens=True,
        max_length=None,
    ):
        content_ids = self.encode(text, add_special_tokens=False)
        ids = [1] + content_ids if add_special_tokens else content_ids
        if truncation and max_length is not None:
            ids = ids[-int(max_length) :]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_qlora_prompt_uses_response_only_label_mask() -> None:
    tokenizer = _ToyTokenizer()
    feature = build_ranker_feature(
        LlamaRecConfig(),
        tokenizer=tokenizer,
        metadata={1: "History One", 2: "Candidate Two", 3: "Target Three"},
        history=[1],
        candidates=[2, 3],
        target=3,
        eval_mode=False,
        original_user_id=7,
    )
    assert feature["prompt"].endswith("### Response: ")
    assert feature["answer_letter"] == "B"
    active_labels = [label for label in feature["labels"] if label != -100]
    assert active_labels == [33, tokenizer.eos_token_id]
    assert feature["labels"][feature["response_start"] - 1] == -100


def test_qlora_prompt_matches_expected_template_character_for_character() -> None:
    prompt = LlamaRecModel(LlamaRecConfig()).build_prompt(
        ["History One", "History Two"],
        ["Candidate One", "Candidate Two"],
    )
    assert prompt == (
        "### Instruction:\n"
        "Given user history in chronological order, recommend an item from the candidate pool with its index letter.\n\n"
        "### Input:\n"
        "User history: (1) History One \n (2) History Two; \n "
        "Candidate pool: (A) Candidate One \n (B) Candidate Two\n\n"
        "### Response: "
    )


def test_qlora_collator_left_truncates_without_losing_response() -> None:
    tokenizer = _ToyTokenizer()
    features = [
        {
            "input_ids": list(range(10)) + [32, 99],
            "attention_mask": [1] * 12,
            "labels": [-100] * 10 + [32, 99],
        },
        {
            "input_ids": [1, 33, 99],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 33, 99],
        },
    ]
    batch = collate_ranker_features(features, max_length=5, pad_token_id=0, eval_mode=False)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"][0].tolist()[-2:] == [32, 99]
    assert batch["labels"][1].tolist() == [-100, -100, -100, 33, 99]
    assert np.isin(batch["labels"].numpy(), [-100, 32, 33, 99]).all()


class _IndexDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return int(index)


def test_use_external_train_dataloader_order_is_reproducible() -> None:
    def collect():
        np.random.seed(42)
        torch.manual_seed(42)
        loader = build_external_train_dataloader(
            _IndexDataset(),
            batch_size=1,
            num_workers=0,
            collate_fn=lambda rows: rows,
        )
        return [batch[0] for batch in loader]

    assert collect() == collect()


def test_verbalizer_ties_preserve_candidate_order() -> None:
    scores = torch.tensor([[2.0, 3.0, 3.0, 1.0], [4.0, 4.0, 5.0, 4.0]])
    positions = rank_verbalizer_scores(scores, k=4, stable_candidate_ties=True)
    assert positions.tolist() == [[1, 2, 0, 3], [2, 0, 1, 3]]