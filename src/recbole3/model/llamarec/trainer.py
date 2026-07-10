from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import pickle
import random
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from recbole3.dataset import CANDIDATE_ITEM_IDS
from recbole3.dataset.ml100k import ORIGINAL_ITEM_ID, ORIGINAL_USER_ID
from recbole3.evaluation.config import EvalConfig, MetricSpec
from recbole3.model.llamarec.config import LlamaRecConfig
from recbole3.model.llamarec.model import LlamaRecModel, rank_verbalizer_scores
from recbole3.trainer_config import TrainerConfig


@dataclass(slots=True)
class LlamaRecTrainerConfig(TrainerConfig):
    run_mode: Literal["sanity", "training", "evaluation", "trace"] = field(default="sanity")
    batch_size: int = field(default=8, metadata={"help": "Effective training batch size."})
    micro_batch_size: int = field(default=1)
    eval_batch_size: int = field(default=8)
    max_epochs: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    warmup_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    early_stopping_patience: int = field(default=20)
    metric_for_best_model: str = field(default="ndcg@10")
    optimizer_name: str = field(default="paged_adamw_32bit")
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=3)
    seed: int = field(default=42)
    sanity_train_samples: int = field(default=3)
    sanity_valid_samples: int = field(default=3)
    sanity_valid_inference_users: int = field(default=5)
    sanity_train_steps: int = field(default=2)
    dataloader_num_workers: int = field(default=4)
    use_external_train_dataloader: bool = field(
        default=True,
        metadata={"help": "Use an external shuffled DataLoader instead of letting HF Trainer rebuild it."},
    )
    trace_max_steps: int = field(
        default=500,
        metadata={"help": "Maximum optimizer steps in trace mode; trace mode never saves an adapter checkpoint."},
    )
    report_to: str = field(default="none")
    candidate_topk: int = field(
        default=20,
        metadata={"help": "Stage-1 candidate count used when auto-generating LlamaRec retrieved.pkl."},
    )
    save_full_score_matrix: bool = field(
        default=False,
        metadata={"help": "Whether auto stage-1 stores full score matrices in retrieved.pkl."},
    )
    eval: EvalConfig = field(
        default_factory=lambda: EvalConfig(
            protocol="full",
            metrics=(
                MetricSpec(name="recall", ks=(1, 5, 10)),
                MetricSpec(name="ndcg", ks=(1, 5, 10)),
                MetricSpec(name="mrr", ks=(1, 5, 10)),
            ),
            neg_sampling_num=0,
            exclude_history=False,
        ),
        kw_only=True,
    )


@dataclass(frozen=True, slots=True)
class LlamaRecRankerRecord:
    original_user_id: int
    history_original_item_ids: tuple[int, ...]
    target_original_item_id: int
    candidate_original_item_ids: tuple[int, ...]
    target_in_candidates: bool


class LlamaRecTrainDataset(Dataset[dict[str, Any]]):
    """Prefix expansion and on-access negative sampling for LlamaRec QLoRA training."""

    def __init__(
        self,
        config: LlamaRecConfig,
        artifact: dict[str, Any],
        tokenizer: Any,
        *,
        seed: int,
    ):
        self.config = config
        self.artifact = artifact
        self.tokenizer = tokenizer
        self.rng = np.random
        self.seed = int(seed)
        self.prefixes: list[tuple[int, tuple[int, ...]]] = []
        for user_id in sorted(int(user_id) for user_id in artifact["train"]):
            sequence = tuple(int(item_id) for item_id in artifact["train"][user_id])
            for end in range(2, len(sequence) + 1):
                self.prefixes.append((user_id, sequence[:end]))

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user_id, tokens = self.prefixes[int(index)]
        target = int(tokens[-1])
        original_history = tuple(int(item_id) for item_id in tokens[:-1])
        history = original_history[-int(self.config.history_max_length) :]
        candidates = [target]
        sample_count = 5 * int(self.config.negative_sample_size)
        samples = self.rng.randint(1, len(self.artifact["smap"]) + 1, size=sample_count)
        cursor = 0
        while len(candidates) < int(self.config.negative_sample_size) + 1:
            if cursor >= len(samples):
                samples = self.rng.randint(1, len(self.artifact["smap"]) + 1, size=sample_count)
                cursor = 0
            item_id = int(samples[cursor])
            cursor += 1
            # Keep duplicate negatives possible to match the original sampling behavior.
            if item_id in original_history or item_id == target:
                continue
            candidates.append(item_id)
        self.rng.shuffle(candidates)
        feature = build_ranker_feature(
            self.config,
            tokenizer=self.tokenizer,
            metadata=self.artifact["meta"],
            history=history,
            candidates=candidates,
            target=target,
            eval_mode=False,
            original_user_id=user_id,
        )
        feature.update(
            sample_index=int(index),
            prefix_original_item_ids=tuple(int(item_id) for item_id in tokens),
            history_original_item_ids=original_history[-int(self.config.history_max_length) :],
            negative_original_item_ids=tuple(
                int(item_id) for item_id in candidates if int(item_id) != target
            ),
            positive_position=int(candidates.index(target)),
        )
        return feature


def train_worker_init_fn(worker_id: int) -> None:
    """Seed NumPy/Python workers deterministically."""
    worker_seed = int(np.random.get_state()[1][0]) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_external_train_dataloader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
    collate_fn: Any,
) -> DataLoader:
    """Build an external shuffled DataLoader for QLoRA training."""
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=True,
        num_workers=int(num_workers),
        worker_init_fn=train_worker_init_fn,
        collate_fn=collate_fn,
    )


class LlamaRecEvalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: LlamaRecConfig,
        artifact: dict[str, Any],
        tokenizer: Any,
        records: Sequence[LlamaRecRankerRecord],
    ):
        self.config = config
        self.artifact = artifact
        self.tokenizer = tokenizer
        self.records = tuple(record for record in records if record.target_in_candidates)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        return build_ranker_feature(
            self.config,
            tokenizer=self.tokenizer,
            metadata=self.artifact["meta"],
            history=record.history_original_item_ids[-int(self.config.history_max_length) :],
            candidates=record.candidate_original_item_ids,
            target=record.target_original_item_id,
            eval_mode=True,
            original_user_id=record.original_user_id,
        )


def load_llamarec_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"LlamaRec dataset.pkl not found at {artifact_path}.")
    with artifact_path.open("rb") as handle:
        artifact = pickle.load(handle)
    required = {"train", "val", "test", "meta", "umap", "smap"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"LlamaRec dataset.pkl is missing keys: {sorted(missing)}")
    return artifact


def build_eval_records(
    artifact: dict[str, Any],
    frame: Any,
    *,
    split: Literal["valid", "test"],
) -> list[LlamaRecRankerRecord]:
    records: list[LlamaRecRankerRecord] = []
    for _, row in frame.iterrows():
        user_id = int(row[ORIGINAL_USER_ID])
        history = list(int(item_id) for item_id in artifact["train"][user_id])
        if split == "test":
            history.extend(int(item_id) for item_id in artifact["val"][user_id])
        candidates = tuple(int(item_id) + 1 for item_id in row[CANDIDATE_ITEM_IDS])
        target = int(row[ORIGINAL_ITEM_ID])
        records.append(
            LlamaRecRankerRecord(
                original_user_id=user_id,
                history_original_item_ids=tuple(history),
                target_original_item_id=target,
                candidate_original_item_ids=candidates,
                target_in_candidates=target in candidates,
            )
        )
    return records


def build_ranker_feature(
    config: LlamaRecConfig,
    *,
    tokenizer: Any,
    metadata: dict[int, str],
    history: Sequence[int],
    candidates: Sequence[int],
    target: int,
    eval_mode: bool,
    original_user_id: int,
) -> dict[str, Any]:
    candidate_ids = tuple(int(item_id) for item_id in candidates)
    if int(target) not in candidate_ids:
        raise ValueError("Ranker features can only be built for retrieval-hit users.")
    prompt_model = LlamaRecModel(config)
    prompt = prompt_model.build_prompt(
        [str(metadata[int(item_id)]) for item_id in history],
        [str(metadata[int(item_id)]) for item_id in candidate_ids],
        tokenizer=tokenizer,
    )
    answer_letter = chr(ord("A") + candidate_ids.index(int(target)))
    prefix = tokenizer(prompt, truncation=False, padding=False, add_special_tokens=True)
    feature: dict[str, Any] = {
        "prompt": prompt,
        "answer_letter": answer_letter,
        "candidate_original_item_ids": candidate_ids,
        "target_original_item_id": int(target),
        "original_user_id": int(original_user_id),
    }
    if eval_mode:
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=int(config.max_text_length),
            padding=False,
            add_special_tokens=True,
        )
        feature.update(
            input_ids=list(tokenized["input_ids"]),
            attention_mask=list(tokenized["attention_mask"]),
            labels=ord(answer_letter) - ord("A"),
        )
        return feature

    answer = tokenizer(answer_letter, truncation=False, padding=False, add_special_tokens=False)
    input_ids = list(prefix["input_ids"]) + list(answer["input_ids"])
    attention_mask = list(prefix["attention_mask"]) + [1] * len(answer["input_ids"])
    if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(int(tokenizer.eos_token_id))
        attention_mask.append(1)
    labels = list(input_ids)
    response_start = len(prefix["input_ids"])
    if not bool(config.train_on_inputs):
        labels[:response_start] = [-100] * response_start
    feature.update(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        response_start=response_start,
    )
    return feature


def collate_ranker_features(
    features: Sequence[dict[str, Any]],
    *,
    max_length: int,
    pad_token_id: int,
    eval_mode: bool,
) -> dict[str, torch.Tensor]:
    width = min(int(max_length), max(len(feature["input_ids"]) for feature in features))
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    label_rows: list[Any] = []
    for feature in features:
        input_ids = list(feature["input_ids"])[-width:]
        attention_mask = list(feature["attention_mask"])[-width:]
        padding = width - len(input_ids)
        input_rows.append([int(pad_token_id)] * padding + input_ids)
        mask_rows.append([0] * padding + attention_mask)
        if eval_mode:
            label_rows.append(int(feature["labels"]))
        else:
            labels = list(feature["labels"])[-width:]
            label_rows.append([-100] * padding + labels)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def load_qwen_qlora_model(config: LlamaRecConfig) -> tuple[Any, Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        trust_remote_code=bool(config.trust_remote_code),
        local_files_only=bool(config.local_files_only),
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(config.use_double_quant),
        bnb_4bit_quant_type=str(config.quant_type),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=bool(config.trust_remote_code),
        local_files_only=bool(config.local_files_only),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        target_modules=list(config.lora_target_modules),
        lora_dropout=float(config.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("LoRA injection produced zero trainable parameters.")
    details = {
        "base_model": str(config.base_model),
        "compute_dtype": str(compute_dtype),
        "load_in_4bit": True,
        "double_quant": bool(config.use_double_quant),
        "quant_type": str(config.quant_type),
        "lora_r": int(config.lora_r),
        "lora_alpha": int(config.lora_alpha),
        "lora_dropout": float(config.lora_dropout),
        "target_modules": list(config.lora_target_modules),
        "trainable_parameters": int(trainable),
    }
    return model, tokenizer, details


class LlamaRecTrainer:
    def __init__(self, model_config: LlamaRecConfig, config: LlamaRecTrainerConfig):
        self.model_config = model_config
        self.config = config

    def run(
        self,
        *,
        mapping_source: str | Path,
        valid_frame: Any,
        test_frame: Any,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed))
        artifact = load_llamarec_artifact(mapping_source)
        valid_records = build_eval_records(artifact, valid_frame, split="valid")
        test_records = build_eval_records(artifact, test_frame, split="test")
        model, tokenizer, qlora_details = load_qwen_qlora_model(self.model_config)
        train_dataset = LlamaRecTrainDataset(
            self.model_config,
            artifact,
            tokenizer,
            seed=int(self.config.seed),
        )
        valid_dataset = LlamaRecEvalDataset(self.model_config, artifact, tokenizer, valid_records)
        test_dataset = LlamaRecEvalDataset(self.model_config, artifact, tokenizer, test_records)
        if self.config.run_mode == "sanity":
            return self._run_sanity(
                model,
                tokenizer,
                train_dataset,
                valid_dataset,
                valid_records,
                qlora_details=qlora_details,
                output_dir=output_path,
            )
        if self.config.run_mode == "evaluation":
            raise ValueError("Evaluation mode requires an explicit trained adapter checkpoint.")
        return self._run_training(
            model,
            tokenizer,
            train_dataset,
            valid_dataset,
            test_dataset,
            valid_records=valid_records,
            test_records=test_records,
            qlora_details=qlora_details,
            output_dir=output_path,
        )

    def _run_sanity(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: LlamaRecTrainDataset,
        valid_dataset: LlamaRecEvalDataset,
        valid_records: Sequence[LlamaRecRankerRecord],
        *,
        qlora_details: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        train_features = [
            train_dataset[index]
            for index in range(min(int(self.config.sanity_train_samples), len(train_dataset)))
        ]
        valid_features = [
            valid_dataset[index]
            for index in range(min(int(self.config.sanity_valid_samples), len(valid_dataset)))
        ]
        prompts_path = output_dir / "sanity_prompts.json"
        prompts_path.write_text(
            json.dumps({"train": train_features, "valid": valid_features}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        train_loader = DataLoader(
            train_features,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda rows: collate_ranker_features(
                rows,
                max_length=int(self.model_config.max_text_length),
                pad_token_id=int(tokenizer.pad_token_id),
                eval_mode=False,
            ),
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(self.config.learning_rate),
        )
        losses: list[float] = []
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= int(self.config.sanity_train_steps):
                break
            batch = {name: tensor.to(_model_input_device(model)) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Sanity loss is not finite at step {step}: {loss.item()}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        inference_count = min(int(self.config.sanity_valid_inference_users), len(valid_dataset))
        rankings = self._rank_eval_features(
            model,
            tokenizer,
            [valid_dataset[index] for index in range(inference_count)],
        )
        letter_ids = _letter_token_ids(tokenizer, int(self.model_config.candidate_topk))
        mask_checks = [_loss_mask_summary(feature, tokenizer) for feature in train_features]
        report = {
            "mode": "sanity",
            "train_dataset_size": len(train_dataset),
            "valid_retrieval_hits": len(valid_dataset),
            "valid_total_users": len(valid_records),
            "constructed_train_samples": len(train_features),
            "constructed_valid_samples": len(valid_features),
            "train_steps": len(losses),
            "losses": losses,
            "losses_finite": all(np.isfinite(losses)),
            "valid_inference_users": inference_count,
            "rankings": rankings,
            "letter_token_ids": letter_ids,
            "loss_mask_checks": mask_checks,
            "qlora": qlora_details,
            "prompt_dump": str(prompts_path),
        }
        report_path = output_dir / "sanity_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _run_training(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset[Any],
        valid_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        *,
        valid_records: Sequence[LlamaRecRankerRecord],
        test_records: Sequence[LlamaRecRankerRecord],
        qlora_details: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

        gradient_accumulation = int(self.config.batch_size) // int(self.config.micro_batch_size)
        if gradient_accumulation <= 0 or int(self.config.batch_size) % int(self.config.micro_batch_size):
            raise ValueError("batch_size must be a positive multiple of micro_batch_size.")
        letter_token_ids = _letter_token_ids(tokenizer, int(self.model_config.candidate_topk))
        startup_manifest = {
            "run_mode": str(self.config.run_mode),
            "train_samples": len(train_dataset),
            "valid_retrieval_hits": len(valid_dataset),
            "valid_total_users": len(valid_records),
            "test_retrieval_hits": len(test_dataset),
            "test_total_users": len(test_records),
            "letter_token_ids": letter_token_ids,
            "qlora": qlora_details,
            "effective_batch_size": int(self.config.batch_size),
            "micro_batch_size": int(self.config.micro_batch_size),
            "gradient_accumulation_steps": gradient_accumulation,
            "eval_steps": int(self.config.eval_steps),
            "save_steps": int(self.config.save_steps),
            "metric_for_best_model": str(self.config.metric_for_best_model),
            "use_external_train_dataloader": bool(self.config.use_external_train_dataloader),
        }
        startup_path = output_dir / "startup_manifest.json"
        startup_path.write_text(json.dumps(startup_manifest, indent=2), encoding="utf-8")
        trace_mode = self.config.run_mode == "trace"
        external_train_dataloader = (
            build_external_train_dataloader(
                train_dataset,
                batch_size=int(self.config.micro_batch_size),
                num_workers=int(self.config.dataloader_num_workers),
                collate_fn=lambda rows: collate_ranker_features(
                    rows,
                    max_length=int(self.model_config.max_text_length),
                    pad_token_id=int(tokenizer.pad_token_id),
                    eval_mode=False,
                ),
            )
            if self.config.use_external_train_dataloader
            else None
        )
        external_total_steps = None
        if external_train_dataloader is not None:
            external_total_steps = (
                len(external_train_dataloader)
                // gradient_accumulation
                * int(self.config.max_epochs)
            )
            startup_manifest["external_total_training_steps"] = int(external_total_steps)
            startup_path.write_text(json.dumps(startup_manifest, indent=2), encoding="utf-8")
        print("LlamaRec QLoRA startup manifest:")
        print(json.dumps(startup_manifest, indent=2))
        kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(self.config.micro_batch_size),
            "per_device_eval_batch_size": int(self.config.eval_batch_size),
            "gradient_accumulation_steps": gradient_accumulation,
            "warmup_steps": int(self.config.warmup_steps),
            "num_train_epochs": int(self.config.max_epochs),
            "learning_rate": float(self.config.learning_rate),
            "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            "logging_steps": int(self.config.logging_steps),
            "optim": str(self.config.optimizer_name),
            "save_strategy": "no" if trace_mode else "steps",
            "eval_steps": int(self.config.eval_steps),
            "save_steps": int(self.config.save_steps),
            "save_total_limit": int(self.config.save_total_limit),
            "load_best_model_at_end": not trace_mode,
            "metric_for_best_model": str(self.config.metric_for_best_model),
            "greater_is_better": True,
            "report_to": [] if self.config.report_to == "none" else [self.config.report_to],
            "dataloader_num_workers": int(self.config.dataloader_num_workers),
            "seed": int(self.config.seed),
        }
        if trace_mode:
            kwargs["max_steps"] = int(self.config.trace_max_steps)
        elif external_total_steps is not None:
            # Match the DataLoader step count used by this training loop.
            kwargs["max_steps"] = int(external_total_steps)
        argument_names = inspect.signature(TrainingArguments.__init__).parameters
        kwargs["eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"] = "steps"
        training_args = TrainingArguments(**kwargs)
        parameter_norm_callback = _build_parameter_norm_callback()
        callbacks: list[Any] = [parameter_norm_callback]
        if not trace_mode:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=int(self.config.early_stopping_patience))
            )
        trainer = _VerbalizerTrainer(
            verbalizer_token_ids=letter_token_ids,
            external_train_dataloader=external_train_dataloader,
            scheduler_num_training_steps=external_total_steps,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=lambda rows: collate_ranker_features(
                rows,
                max_length=int(self.model_config.max_text_length),
                pad_token_id=int(tokenizer.pad_token_id),
                eval_mode=bool(rows and isinstance(rows[0]["labels"], (int, np.integer))),
            ),
            compute_metrics=_compute_subset_metrics,
            callbacks=callbacks,
            **_trainer_tokenizer_kwargs(Trainer, tokenizer),
        )
        trainer.train()
        if trace_mode:
            trace = {
                "mode": "trace",
                "max_steps": int(self.config.trace_max_steps),
                "use_external_train_dataloader": bool(self.config.use_external_train_dataloader),
                "trainer_log_history": trainer.state.log_history,
                "parameter_norm_history": parameter_norm_callback.records,
                "qlora": qlora_details,
            }
            (output_dir / "short_training_trace.json").write_text(
                json.dumps(trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return trace
        trainer.save_model(str(output_dir / "best_adapter"))
        tokenizer.save_pretrained(str(output_dir / "best_adapter"))
        model.eval()
        valid_rankings = self._rank_eval_features(
            model,
            tokenizer,
            [valid_dataset[index] for index in range(len(valid_dataset))],
        )
        test_rankings = self._rank_eval_features(
            model,
            tokenizer,
            [test_dataset[index] for index in range(len(test_dataset))],
        )
        valid_metrics = _ranking_metrics(valid_rankings, total_users=len(valid_records))
        test_metrics = _ranking_metrics(test_rankings, total_users=len(test_records))
        result = {
            "mode": "training",
            "qlora": qlora_details,
            "valid_retrieval_hits": len(valid_dataset),
            "valid_total_users": len(valid_records),
            "test_retrieval_hits": len(test_dataset),
            "test_total_users": len(test_records),
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "best_adapter": str(output_dir / "best_adapter"),
        }
        (output_dir / "final_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    def _rank_eval_features(
        self,
        model: Any,
        tokenizer: Any,
        features: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not features:
            return []
        loader = DataLoader(
            features,
            batch_size=int(self.config.eval_batch_size),
            shuffle=False,
            collate_fn=lambda rows: collate_ranker_features(
                rows,
                max_length=int(self.model_config.max_text_length),
                pad_token_id=int(tokenizer.pad_token_id),
                eval_mode=True,
            ),
        )
        letter_ids = torch.tensor(
            _letter_token_ids(tokenizer, int(self.model_config.candidate_topk)),
            dtype=torch.long,
            device=_model_input_device(model),
        )
        results: list[dict[str, Any]] = []
        offset = 0
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                batch = {name: tensor.to(_model_input_device(model)) for name, tensor in batch.items()}
                logits = model(**batch).logits[:, -1, :].float().index_select(1, letter_ids)
                order = rank_verbalizer_scores(logits, k=logits.shape[1], stable_candidate_ties=True).cpu()
                for row_index, positions in enumerate(order.tolist()):
                    feature = features[offset + row_index]
                    candidates = list(feature["candidate_original_item_ids"])
                    results.append(
                        {
                            "original_user_id": int(feature["original_user_id"]),
                            "target_letter_index": int(labels[row_index]),
                            "target_original_item_id": int(feature["target_original_item_id"]),
                            "reranked_original_item_ids": [int(candidates[position]) for position in positions],
                        }
                    )
                offset += len(labels)
        return results


class _VerbalizerTrainer:
    def __new__(
        cls,
        *,
        verbalizer_token_ids: Sequence[int],
        external_train_dataloader: DataLoader | None = None,
        scheduler_num_training_steps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        from transformers import Trainer

        class Impl(Trainer):
            def get_train_dataloader(self):
                if external_train_dataloader is not None:
                    return external_train_dataloader
                return super().get_train_dataloader()

            def create_scheduler(self, num_training_steps, optimizer=None):
                if scheduler_num_training_steps is not None:
                    num_training_steps = int(scheduler_num_training_steps)
                return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop("labels").detach()
                if prediction_loss_only:
                    return None, None, None
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :].float()
                    ids = torch.tensor(verbalizer_token_ids, dtype=torch.long, device=logits.device)
                    scores = logits.index_select(1, ids)
                return None, scores.detach(), labels

        return Impl(**kwargs)


def _compute_subset_metrics(eval_prediction: Any) -> dict[str, float]:
    scores = torch.as_tensor(eval_prediction.predictions)
    labels = torch.as_tensor(eval_prediction.label_ids).long().view(-1)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    matches = order.eq(labels.unsqueeze(1))
    ranks = torch.argmax(matches.long(), dim=1) + 1
    results: dict[str, float] = {
        "verbalizer_logit_mean": float(scores.float().mean()),
        "verbalizer_logit_std": float(scores.float().std(unbiased=False)),
        "target_rank_mean": float(ranks.float().mean()),
    }
    for k in (1, 5, 10):
        hit = matches[:, :k].any(dim=1)
        results[f"recall@{k}"] = float(hit.float().mean())
        results[f"ndcg@{k}"] = float(torch.where(hit, 1.0 / torch.log2(ranks.float() + 1), 0.0).mean())
        results[f"mrr@{k}"] = float(torch.where(hit, 1.0 / ranks.float(), 0.0).mean())
    return results


def _build_parameter_norm_callback() -> Any:
    from transformers import TrainerCallback

    class ParameterNormCallback(TrainerCallback):
        def __init__(self) -> None:
            self.records: list[dict[str, float | int]] = []

        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            if model is None or not logs or "loss" not in logs:
                return control
            squared = torch.zeros((), dtype=torch.float64)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        squared += parameter.detach().float().norm().double().pow(2).cpu()
            record: dict[str, float | int] = {
                "step": int(state.global_step),
                "parameter_norm": float(torch.sqrt(squared)),
            }
            if "grad_norm" in logs:
                record["grad_norm"] = float(logs["grad_norm"])
            self.records.append(record)
            return control

    return ParameterNormCallback()


def _ranking_metrics(rankings: Sequence[dict[str, Any]], *, total_users: int) -> dict[str, Any]:
    subset_size = len(rankings)
    subset: dict[str, float] = {}
    for k in (1, 5, 10):
        recall_values: list[float] = []
        ndcg_values: list[float] = []
        mrr_values: list[float] = []
        for ranking in rankings:
            ordered = ranking["reranked_original_item_ids"]
            target = int(ranking["target_original_item_id"])
            rank = ordered.index(target) + 1 if target in ordered else None
            hit = rank is not None and rank <= k
            recall_values.append(float(hit))
            ndcg_values.append(1.0 / np.log2(rank + 1) if hit else 0.0)
            mrr_values.append(1.0 / rank if hit else 0.0)
        subset[f"recall@{k}"] = float(np.mean(recall_values)) if recall_values else 0.0
        subset[f"ndcg@{k}"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
        subset[f"mrr@{k}"] = float(np.mean(mrr_values)) if mrr_values else 0.0
    ratio = subset_size / int(total_users) if int(total_users) else 0.0
    return {
        "subset": subset,
        "overall": {name: value * ratio for name, value in subset.items()},
        "retrieval_hits": subset_size,
        "total_users": int(total_users),
        "top20_hit_ratio": ratio,
    }


def _letter_token_ids(tokenizer: Any, count: int) -> list[int]:
    ids: list[int] = []
    for index in range(int(count)):
        letter = chr(ord("A") + index)
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Bare verbalizer {letter!r} must encode to one token, got {encoded}.")
        ids.append(int(encoded[0]))
    return ids


def _loss_mask_summary(feature: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    active = [index for index, label in enumerate(feature["labels"]) if int(label) != -100]
    active_ids = [int(feature["labels"][index]) for index in active]
    expected = tokenizer.encode(feature["answer_letter"], add_special_tokens=False) + [int(tokenizer.eos_token_id)]
    return {
        "original_user_id": int(feature["original_user_id"]),
        "active_positions": active,
        "active_token_ids": active_ids,
        "expected_token_ids": expected,
        "matches": active_ids == expected,
    }


def _model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _trainer_tokenizer_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


__all__ = [
    "LlamaRecEvalDataset",
    "LlamaRecTrainer",
    "LlamaRecTrainerConfig",
    "LlamaRecRankerRecord",
    "LlamaRecTrainDataset",
    "build_eval_records",
    "build_ranker_feature",
    "build_external_train_dataloader",
    "collate_ranker_features",
    "load_llamarec_artifact",
    "load_qwen_qlora_model",
    "train_worker_init_fn",
]
