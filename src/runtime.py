"""Device / dtype / model loading, shared by training, inference and export.

Keeping this in one place is what makes "runs on a laptop and on an A10"
true rather than aspirational: every entry point resolves hardware the same way.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from src.config import ModelConfig

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def resolve_device(requested: str | None = "auto") -> str:
    if requested and requested not in ("auto", "null", "none"):
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_bf16_supported()
    return False  # MPS bf16 is unreliable for training; CPU bf16 is slow


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested and requested != "auto":
        return _DTYPES[requested.lower()]
    if device == "cuda":
        return torch.bfloat16 if supports_bf16(device) else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def quantization_config(cfg: ModelConfig, device: str):
    """BitsAndBytesConfig for QLoRA, or None when unavailable/disabled."""
    if not cfg.quantization.enabled:
        return None
    if not bitsandbytes_available():
        warnings.warn(
            "quantization.enabled is true but bitsandbytes is not usable here "
            f"(device={device}). Falling back to a non-quantized {cfg.dtype} load; "
            "LoRA still works, it just uses more memory.",
            stacklevel=2,
        )
        return None
    from transformers import BitsAndBytesConfig

    q = cfg.quantization
    return BitsAndBytesConfig(
        load_in_4bit=q.load_in_4bit,
        bnb_4bit_quant_type=q.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_DTYPES[q.bnb_4bit_compute_dtype.lower()],
        bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
    )


def load_tokenizer(cfg: ModelConfig, path: str | Path | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(path or cfg.resolved_base_model()),
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training; generation flips it explicitly
    return tokenizer


def load_base_model(cfg: ModelConfig, device: str | None = None, for_training: bool = False):
    from transformers import AutoModelForCausalLM

    device = device or resolve_device(cfg.device_map)
    dtype = resolve_dtype(cfg.dtype, device)
    if for_training and device == "mps" and cfg.dtype == "auto":
        # fp16 training on MPS produces NaNs in practice; fp32 is slower but works.
        dtype = torch.float32
    quant = quantization_config(cfg, device)

    kwargs: dict = {
        "dtype": dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "revision": cfg.revision,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if quant is not None:
        kwargs["quantization_config"] = quant
        kwargs["device_map"] = {"": 0}
    elif device == "cuda" and not for_training:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.resolved_base_model(), **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.config.use_cache = not for_training
    return model, device, dtype
