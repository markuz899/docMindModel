#!/usr/bin/env python
"""Build a ~2M-parameter Qwen2 in artifacts/tiny-model, fully offline.

Smoke tests need a model, not a good model. This one is randomly initialised
and its output is gibberish by design -- what it proves is that the training,
inference, evaluation and export code paths actually run, without a 3 GB
download or a network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def corpus_text() -> list[str]:
    root = resolve_path("data")
    texts = [p.read_text(encoding="utf-8") for p in root.rglob("*.md")]
    texts += [p.read_text(encoding="utf-8") for p in root.rglob("*.jsonl")]
    texts += [p.read_text(encoding="utf-8") for p in resolve_path("src").rglob("*.py")]
    return texts or ["the quick brown fox jumps over the lazy dog"]


def build_tokenizer(out_dir: Path, vocab_size: int = 4096):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
    tok = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials, min_frequency=2)
    tok.train_from_iterator(corpus_text(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        bos_token="<|im_start|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|endoftext|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.model_max_length = 2048
    fast.save_pretrained(out_dir)
    return fast


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    out_dir = resolve_path("artifacts/tiny-model")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(out_dir)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(out_dir, safe_serialization=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"tiny model: {params/1e6:.2f}M params, vocab {len(tokenizer)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
