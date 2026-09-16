"""YAML configs -> validated pydantic objects.

Single entry point for every script, so retargeting the pipeline to another
base model is a one-line change in configs/model.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class QuantizationConfig(_Base):
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class LoraSettings(_Base):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    modules_to_save: list[str] = Field(default_factory=list)


class GenerationSettings(_Base):
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05


class ExportSettings(_Base):
    """GGUF export defaults. Q4_K_M is what ships with the DocMind app."""

    merged_dir: str = "artifacts/export/merged"
    gguf_dir: str = "artifacts/export/gguf"
    outtype: str = "f16"                      # intermediate GGUF precision
    quantizations: list[str] = Field(default_factory=lambda: ["Q4_K_M", "Q5_K_M", "Q8_0"])
    primary_quantization: str = "Q4_K_M"


class ModelConfig(_Base):
    base_model: str
    revision: str | None = None
    trust_remote_code: bool = False
    max_seq_length: int = 4096
    dtype: str = "auto"
    device_map: str | None = "auto"
    attn_implementation: str | None = None
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)

    def resolved_base_model(self) -> str:
        """Local paths are resolved against the repo root; hub ids pass through."""
        candidate = (REPO_ROOT / self.base_model)
        return str(candidate) if candidate.exists() else self.base_model


class TrainingConfig(_Base):
    output_dir: str = "artifacts/adapter"
    run_name: str = "docmind-lora"
    seed: int = 42

    num_train_epochs: float = 3
    max_steps: int = -1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True

    bf16: bool | Literal["auto"] = "auto"
    fp16: bool | Literal["auto"] = "auto"

    logging_steps: int = 5
    eval_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    resume_from_checkpoint: str | None = None

    completion_only_loss: bool = True
    skip_overlong: bool = True
    # Training-only context cap. The backward pass is what runs out of memory,
    # not generation, so this can be lower than the model's own max_seq_length
    # without shrinking the context the model is served at inference time.
    max_seq_length: int | None = None
    empty_cache_steps: int = 1  # 0 disables; matters on MPS unified memory
    dataloader_num_workers: int = 0
    report_to: list[str] = Field(default_factory=list)


class SplitConfig(_Base):
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1
    group_by: Literal["project", "document", "none"] = "project"
    near_duplicate_threshold: float = 0.9


class DedupConfig(_Base):
    enabled: bool = True
    question_similarity: float = 0.82   # token-Jaccard above this = duplicate
    per_context_limit: int = 6          # keep >= generation.examples_per_teacher_call


class QualityConfig(_Base):
    min_question_chars: int = 10
    max_question_chars: int = 500
    min_answer_chars: int = 20
    max_answer_chars: int = 4000
    max_answer_context_ratio: float = 1.5
    long_answer_min_chars: int = 1200
    min_groundedness: float = 0.15
    min_technical_groundedness: float = 0.75
    max_unsupported_identifiers: int = 0
    max_unsupported_identifiers_refusal: int = 3
    require_citation_when_answerable: bool = True


class ChunkingConfig(_Base):
    min_heading_level: int = 1
    max_heading_level: int = 4
    min_chunk_chars: int = 80
    max_chunk_chars: int = 4000


class AnswerabilityMix(_Base):
    """Target shape of the generated dataset. Normalised before use."""

    full: float = 0.55
    partial: float = 0.15
    none: float = 0.20
    noisy: float = 0.10

    def weights(self) -> dict[str, float]:
        raw = {"full": self.full, "partial": self.partial,
               "none": self.none, "noisy": self.noisy}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


class GenerationPipelineConfig(_Base):
    # `auto` prefers a local, subscription-backed CLI and never falls back to a
    # metered API. See src/generation/providers.py: select_teacher.
    teacher: Literal["auto", "codex", "claude", "mock", "openai", "anthropic"] = "auto"
    model: str | None = None
    timeout_s: int = 300
    temperature: float = 0.4
    max_output_tokens: int = 4000

    # Batching: one CLI invocation should yield several examples. 3000 examples
    # at 5 per call is ~600 subprocesses, not 3000.
    examples_per_teacher_call: int = 5
    max_teacher_calls: int | None = None
    concurrency: int = 1

    cache_enabled: bool = True
    cache_dir: str = "data/cache/teacher"

    answerability: AnswerabilityMix = Field(default_factory=AnswerabilityMix)
    context_sizes: list[int] = Field(default_factory=lambda: [2, 3, 4, 5])
    distractor_ratio: float = 0.35
    contradiction_ratio: float = 0.08
    max_contexts: int = 600
    seed: int = 7

    # --- retained for the demo pipeline / older configs ---
    provider: Literal["mock", "openai", "anthropic", "codex", "claude"] = "mock"
    questions_per_context: int = 2
    unanswerable_ratio: float = 0.25


class DatasetConfig(_Base):
    raw_dir: str = "data/raw"
    generated_dir: str = "data/generated"
    train_path: str = "data/train/train.jsonl"
    validation_path: str = "data/validation/validation.jsonl"
    test_path: str = "data/test/test.jsonl"
    version: str = "0.1.0"
    split: SplitConfig = Field(default_factory=SplitConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    generation: GenerationPipelineConfig = Field(default_factory=GenerationPipelineConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model_config(path: str | Path = "configs/model.yaml") -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_training_config(path: str | Path = "configs/training.yaml") -> TrainingConfig:
    return TrainingConfig(**load_yaml(path))


def load_dataset_config(path: str | Path = "configs/dataset.yaml") -> DatasetConfig:
    return DatasetConfig(**load_yaml(path))


def resolve_path(path: str | Path) -> Path:
    """Repo-root relative path -> absolute Path (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
