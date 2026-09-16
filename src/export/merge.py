#!/usr/bin/env python
"""Merge a LoRA adapter into the base weights and save a standalone model.

    python -m src.export.merge --adapter artifacts/adapter

GGUF conversion cannot read a PEFT adapter, so this step is mandatory before
export. The merge always runs unquantized on CPU: merging into 4-bit weights
loses the adapter's precision.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from src.config import load_model_config, resolve_path
from src.runtime import load_tokenizer


def merge_adapter(model_config: str, adapter: str, out_dir: str | None = None,
                  dtype: str = "float16") -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    cfg = load_model_config(model_config)
    target = resolve_path(out_dir or cfg.export.merged_dir)
    target.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]

    print(f"[merge] loading base {cfg.base_model} on cpu ({dtype}, unquantized)")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.resolved_base_model(),
        dtype=torch_dtype,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
    )
    adapter_path = resolve_path(adapter)
    print(f"[merge] applying adapter {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.config.use_cache = True

    model.save_pretrained(str(target), safe_serialization=True)
    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").exists() else None
    load_tokenizer(cfg, tokenizer_source).save_pretrained(str(target))

    training_metrics = adapter_path / "training_metrics.json"
    metadata = {
        "name": "docmind-model",
        "version": "0.1.0",
        "base_model": cfg.base_model,
        "adapter": str(adapter_path),
        "merged_dtype": dtype,
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": json.loads(training_metrics.read_text(encoding="utf-8"))
        if training_metrics.exists()
        else None,
    }
    (target / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    size_mb = sum(f.stat().st_size for f in target.glob("*.safetensors")) / 1e6
    print(f"[merge] wrote {target} ({size_mb:.0f} MB of weights)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default="artifacts/adapter")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    merge_adapter(args.model_config, args.adapter, args.out, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
