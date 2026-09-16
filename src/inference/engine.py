"""Load a (possibly LoRA-adapted) model and answer grounded questions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from src.config import ModelConfig, load_model_config, resolve_path
from src.prompt import SYSTEM_PROMPT, build_messages
from src.runtime import load_base_model, load_tokenizer, resolve_device


class DocMindEngine:
    def __init__(self, model, tokenizer, device: str, cfg: ModelConfig, label: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = cfg
        self.label = label
        self.model.eval()

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_config: str | Path | ModelConfig = "configs/model.yaml",
        adapter: str | Path | None = None,
        label: str | None = None,
    ) -> "DocMindEngine":
        cfg = (
            model_config
            if isinstance(model_config, ModelConfig)
            else load_model_config(model_config)
        )
        model, device, _ = load_base_model(cfg, for_training=False)
        name = label or cfg.base_model
        if adapter:
            from peft import PeftModel

            adapter_path = resolve_path(adapter)
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.to(device)
            name = label or f"{cfg.base_model}+{adapter_path.name}"
        tokenizer = load_tokenizer(cfg, _tokenizer_dir(adapter) or None)
        return cls(model, tokenizer, device, cfg, name)

    # -- generation ------------------------------------------------------

    def render_prompt(
        self,
        question: str,
        context: Iterable[Mapping[str, str]],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        messages = build_messages(question, context, system_prompt=system_prompt)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        context: Iterable[Mapping[str, str]],
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int | None = None,
    ) -> str:
        return self.generate_batch(
            [(question, list(context))], system_prompt, max_new_tokens
        )[0]

    @torch.inference_mode()
    def generate_batch(
        self,
        items: Sequence[tuple[str, Sequence[Mapping[str, str]]]],
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int | None = None,
        batch_size: int = 1,
    ) -> list[str]:
        gen = self.cfg.generation
        out: list[str] = []
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # decoder-only batching
        try:
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                prompts = [self.render_prompt(q, c, system_prompt) for q, c in chunk]
                enc = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_seq_length - (max_new_tokens or gen.max_new_tokens),
                    add_special_tokens=False,
                ).to(self.device)
                sampling = gen.temperature > 0
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens or gen.max_new_tokens,
                    do_sample=sampling,
                    temperature=gen.temperature if sampling else None,
                    top_p=gen.top_p if sampling else None,
                    repetition_penalty=gen.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                prompt_len = enc["input_ids"].shape[1]
                for row in generated:
                    text = self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                    out.append(text.strip())
        finally:
            self.tokenizer.padding_side = original_side
        return out


def _tokenizer_dir(adapter: str | Path | None) -> Path | None:
    """Adapters saved by this repo carry their tokenizer; prefer it if present."""
    if not adapter:
        return None
    path = resolve_path(adapter)
    return path if (path / "tokenizer_config.json").exists() else None
