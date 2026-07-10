from __future__ import annotations

from dataclasses import dataclass, field

from recbole3.model.sequential import SequentialModelConfig


@dataclass(slots=True)
class LlamaRecConfig(SequentialModelConfig):
    name: str = field(default="llamarec", metadata={"help": "Trainable LlamaRec QLoRA ranker."})
    history_max_length: int = field(default=20, metadata={"help": "Maximum history titles included in the prompt."})
    candidate_topk: int = field(default=20, metadata={"help": "Number of LRURec candidates represented by A-T."})
    retrieved_path: str = field(
        default="outputs/lrurec_grid/wd0_drop0.2/retrieved.pkl",
        metadata={"help": "RecBole3 LRURec candidates used by validation and test."},
    )
    item_text_field: str = field(default="title", metadata={"help": "Item table field used in prompts."})
    base_model: str = field(
        default="/media/public/models/huggingface/Qwen/Qwen2.5-3B",
        metadata={"help": "Local base model path."},
    )
    tokenizer_path: str = field(
        default="/media/public/models/huggingface/Qwen/Qwen2.5-3B",
        metadata={"help": "Local tokenizer path."},
    )
    max_title_tokens: int = field(default=32, metadata={"help": "Maximum tokenizer tokens retained per title."})
    max_text_length: int = field(default=1536, metadata={"help": "Maximum left-truncated prompt length."})
    auto_stage1: bool = field(
        default=True,
        metadata={"help": "Whether to auto-run LRURec stage-1 when retrieved_path is missing."},
    )
    stage1_output_dir: str = field(
        default="",
        metadata={"help": "Optional output directory for auto-generated LRURec stage-1 artifacts."},
    )
    stage1_save_full_score_matrix: bool = field(
        default=False,
        metadata={"help": "Whether Amazon-style auto stage-1 stores full score matrices in retrieved.pkl."},
    )
    stage1_max_epochs: int | None = field(
        default=None,
        metadata={"help": "Optional max_epochs override for auto LRURec stage-1."},
    )
    stage1_history_max_length: int = field(
        default=50,
        metadata={"help": "LRURec history length used by auto stage-1 for generated LlamaRec artifacts."},
    )
    negative_sample_size: int = field(default=19, metadata={"help": "Random train negatives per positive."})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether prompt tokens participate in loss."})
    load_in_4bit: bool = field(default=True, metadata={"help": "Load the base model with NF4 quantization."})
    local_files_only: bool = field(default=True)
    use_double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: tuple[str, ...] = field(default=("q_proj", "v_proj"))
    stable_candidate_ties: bool = field(
        default=True,
        metadata={"help": "Break equal verbalizer logits by original candidate position."},
    )
    trust_remote_code: bool = field(default=True)
    system_template: str = field(
        default="Given user history in chronological order, recommend an item from the candidate pool with its index letter.",
        metadata={"help": "LlamaRec instruction."},
    )
    input_template: str = field(
        default="User history: {}; \n Candidate pool: {}",
        metadata={"help": "LlamaRec prompt input template."},
    )


__all__ = ["LlamaRecConfig"]