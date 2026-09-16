#!/usr/bin/env python
"""LoRA / QLoRA supervised fine-tuning.

    python -m src.training.train
    python -m src.training.train --smoke          # 4 steps on a tiny local model
    python -m src.training.train --resume latest

Memory profile by design: LoRA (optionally over a 4-bit base), gradient
accumulation, gradient checkpointing and mixed precision where the device
supports it. Nothing here assumes a datacentre GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import (
    ModelConfig,
    TrainingConfig,
    load_dataset_config,
    load_model_config,
    load_training_config,
    resolve_path,
)
from src.dataset.io import read_jsonl
from src.runtime import bitsandbytes_available, load_base_model, load_tokenizer, supports_bf16
from src.training.data import CompletionCollator, encode_dataset


def build_peft_model(model, cfg: ModelConfig, quantized: bool, gradient_checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=getattr(TaskType, cfg.lora.task_type),
        target_modules=cfg.lora.target_modules,
        modules_to_save=cfg.lora.modules_to_save or None,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


class EmptyCacheCallback:
    """Release the accelerator's cached blocks every N steps.

    On MPS the caching allocator grows monotonically during a long run; on an
    8-16 GB unified-memory machine that turns into swap thrashing and steps get
    steadily slower (33s -> 47s -> 58s in a measured run here). Dropping the
    cache periodically costs a little re-allocation and keeps the step time flat.
    """

    def __init__(self, device: str, every: int = 1) -> None:
        self.device, self.every = device, max(1, every)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every:
            return control
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        return control

    def __getattr__(self, name):  # satisfy TrainerCallback's event protocol
        if name.startswith("on_"):
            return lambda *a, **k: k.get("control")
        raise AttributeError(name)


def resolve_precision(train_cfg: TrainingConfig, device: str) -> tuple[bool, bool]:
    bf16, fp16 = train_cfg.bf16, train_cfg.fp16
    if bf16 == "auto":
        bf16 = device == "cuda" and supports_bf16(device)
    if fp16 == "auto":
        fp16 = device == "cuda" and not bf16
    return bool(bf16), bool(fp16)


def resolve_optimizer(name: str) -> str:
    if "bnb" in name and not bitsandbytes_available():
        print(f"[train] optim={name} needs bitsandbytes; falling back to adamw_torch")
        return "adamw_torch"
    return name


def find_last_checkpoint(output_dir: Path) -> str | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="use the *.smoke.yaml configs and cap the dataset")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override training.max_steps (memory/time probe)")
    parser.add_argument("--resume", default=None, help="checkpoint path, or 'latest'")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.smoke:
        if args.model_config == "configs/model.yaml":
            args.model_config = "configs/model.smoke.yaml"
        if args.training_config == "configs/training.yaml":
            args.training_config = "configs/training.smoke.yaml"
        args.max_train_examples = args.max_train_examples or 8

    model_cfg = load_model_config(args.model_config)
    train_cfg = load_training_config(args.training_config)
    if args.max_steps is not None:
        train_cfg = train_cfg.model_copy(
            update={"max_steps": args.max_steps, "save_strategy": "no",
                    "eval_strategy": "no", "load_best_model_at_end": False}
        )
    data_cfg = load_dataset_config(args.dataset_config)
    output_dir = resolve_path(args.output_dir or train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(train_cfg.seed)

    train_examples = read_jsonl(data_cfg.train_path)
    eval_examples = read_jsonl(data_cfg.validation_path)
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
        eval_examples = eval_examples[: max(2, args.max_train_examples // 4)]
    if not train_examples:
        raise SystemExit(f"no training data in {data_cfg.train_path} "
                         "-- run scripts/build_dataset.py first")

    tokenizer = load_tokenizer(model_cfg)
    model, device, dtype = load_base_model(model_cfg, for_training=True)
    quantized = getattr(model, "is_quantized", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    print(f"[train] device={device} dtype={dtype} quantized={quantized}")

    gradient_checkpointing = train_cfg.gradient_checkpointing
    model = build_peft_model(model, model_cfg, quantized, gradient_checkpointing)

    max_seq_length = train_cfg.max_seq_length or model_cfg.max_seq_length
    if max_seq_length != model_cfg.max_seq_length:
        print(f"[train] training context capped at {max_seq_length} tokens "
              f"(model serves {model_cfg.max_seq_length})")
    train_encoded, train_stats = encode_dataset(
        train_examples, tokenizer, max_seq_length,
        train_cfg.completion_only_loss, train_cfg.skip_overlong,
    )
    eval_encoded, eval_stats = encode_dataset(
        eval_examples, tokenizer, max_seq_length,
        train_cfg.completion_only_loss, train_cfg.skip_overlong,
    )
    if not train_encoded:
        raise SystemExit(
            f"every training example exceeds max_seq_length={max_seq_length}; "
            "raise it or set skip_overlong: false"
        )
    print(f"[train] train tokens: {train_stats}")
    print(f"[train] eval  tokens: {eval_stats}")

    to_rows = lambda items: [{"input_ids": e.input_ids, "labels": e.labels} for e in items]

    effective_batch = train_cfg.per_device_train_batch_size * train_cfg.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_encoded) / effective_batch))
    total_steps = (
        train_cfg.max_steps
        if train_cfg.max_steps and train_cfg.max_steps > 0
        else steps_per_epoch * int(math.ceil(train_cfg.num_train_epochs))
    )
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    bf16, fp16 = resolve_precision(train_cfg, device)
    eval_strategy = train_cfg.eval_strategy if eval_encoded else "no"
    print(f"[train] effective batch {effective_batch}, ~{total_steps} steps, "
          f"warmup {warmup_steps}, bf16={bf16} fp16={fp16}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=train_cfg.run_name,
        seed=train_cfg.seed,
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps if train_cfg.max_steps and train_cfg.max_steps > 0 else -1,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        optim=resolve_optimizer(train_cfg.optim),
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        bf16=bf16,
        fp16=fp16,
        logging_steps=train_cfg.logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=train_cfg.save_strategy,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end and eval_strategy != "no"
        and train_cfg.save_strategy == eval_strategy,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        report_to=train_cfg.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        use_cpu=device == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=to_rows(train_encoded),
        eval_dataset=to_rows(eval_encoded) if eval_encoded else None,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[EmptyCacheCallback(device, every=train_cfg.empty_cache_steps)]
        if train_cfg.empty_cache_steps
        else None,
    )

    resume = args.resume or train_cfg.resume_from_checkpoint
    if resume == "latest":
        resume = find_last_checkpoint(output_dir)
        print(f"[train] resuming from {resume or 'scratch (no checkpoint found)'}")

    try:
        result = trainer.train(resume_from_checkpoint=resume)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        raise SystemExit(
            f"\nOut of memory on {device}.\n\n"
            f"{exc}\n\n"
            "The usual levers, cheapest first:\n"
            "  1. gradient_checkpointing: true      (configs/training*.yaml)\n"
            "  2. lower max_seq_length              (configs/model*.yaml) -- the loss\n"
            "     tensor is seq_len x vocab_size, which dominates for large vocabularies\n"
            "  3. per_device_train_batch_size: 1 and raise gradient_accumulation_steps\n"
            "  4. quantization.enabled: true        (QLoRA; needs bitsandbytes + CUDA)\n"
            + ("  5. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 lifts the MPS allocation cap\n"
               "     (can destabilise the machine -- use last)\n" if device == "mps" else "")
        ) from exc

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(result.metrics)
    if eval_encoded:
        metrics.update(trainer.evaluate())
    metadata = {
        "name": "docmind-model",
        "version": "0.1.0",
        "base_model": model_cfg.base_model,
        "adapter_type": "qlora" if quantized else "lora",
        "quantization": model_cfg.quantization.model_dump() if quantized else None,
        "lora": model_cfg.lora.model_dump(),
        "max_seq_length": model_cfg.max_seq_length,
        "train_max_seq_length": max_seq_length,
        "training_dataset": data_cfg.train_path,
        "dataset_version": data_cfg.version,
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device,
        "dtype": str(dtype),
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "hyperparameters": train_cfg.model_dump(),
        "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[train] adapter + metrics written to {output_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
