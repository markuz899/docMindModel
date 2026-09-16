"""Examples -> tokenised training tensors, with the prompt masked out of the loss.

Only the assistant turn contributes to the loss. Training on the prompt as well
teaches the model to reproduce documentation blocks, which is the opposite of
what this fine-tune is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.dataset.schema import Example
from src.prompt import SYSTEM_PROMPT, build_messages


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)


def encode_example(
    example: Example,
    tokenizer,
    max_seq_length: int,
    system_prompt: str = SYSTEM_PROMPT,
    completion_only: bool = True,
) -> EncodedExample:
    context = [c.model_dump() for c in example.context]
    prompt_messages = build_messages(example.question, context, system_prompt=system_prompt)
    full_messages = build_messages(
        example.question, context, answer=example.answer, system_prompt=system_prompt
    )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "chat template is not prefix-stable: the prompt rendering is not a "
            "prefix of the full conversation, so the completion cannot be masked "
            "reliably. Check the tokenizer's chat_template."
        )
    completion_text = full_text[len(prompt_text) :]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    # Keep the answer whole and drop the oldest context if the pair is too long.
    budget = max_seq_length - len(completion_ids)
    if budget < 1:
        completion_ids = completion_ids[: max_seq_length - 1]
        budget = 1
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + completion_ids
    labels = (
        [-100] * len(prompt_ids) + list(completion_ids)
        if completion_only
        else list(input_ids)
    )
    return EncodedExample(input_ids=input_ids, labels=labels)


_NO_LIMIT = 10**9


def encode_dataset(
    examples: list[Example],
    tokenizer,
    max_seq_length: int,
    completion_only: bool = True,
    skip_overlong: bool = True,
) -> tuple[list[EncodedExample], dict]:
    """Encode a dataset, dropping (not truncating) examples that do not fit.

    Truncating the prompt removes the *oldest* context chunks -- which may be
    exactly the sections the answer cites. Training on that teaches the model to
    cite sources it cannot see, the precise failure this fine-tune exists to
    prevent. Dropping is the safe default; `skip_overlong=False` restores
    truncation for cases where losing examples is worse.
    """
    encoded, skipped, truncated = [], [], 0
    for ex in examples:
        item = encode_example(ex, tokenizer, _NO_LIMIT, completion_only=completion_only)
        if len(item) > max_seq_length:
            if skip_overlong:
                skipped.append(ex.id)
                continue
            item = encode_example(ex, tokenizer, max_seq_length, completion_only=completion_only)
            truncated += 1
        encoded.append(item)
    if skipped:
        print(
            f"[data] dropped {len(skipped)}/{len(examples)} example(s) longer than "
            f"max_seq_length={max_seq_length}; raise it to keep them "
            f"(first: {', '.join(skipped[:3])})"
        )
    lengths = [len(e) for e in encoded]
    stats = {
        "examples": len(encoded),
        "skipped_overlong": len(skipped),
        "truncated": truncated,
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": round(sum(lengths) / max(len(lengths), 1), 1),
        "trainable_tokens": sum(sum(1 for t in e.labels if t != -100) for e in encoded),
    }
    return encoded, stats


class CompletionCollator:
    """Right-pad a batch; padded positions and the prompt are ignored by the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[EncodedExample | dict]) -> dict[str, torch.Tensor]:
        items = [
            b if isinstance(b, EncodedExample) else EncodedExample(b["input_ids"], b["labels"])
            for b in batch
        ]
        width = max(len(i) for i in items)
        input_ids, labels, attention = [], [], []
        for item in items:
            pad = width - len(item)
            input_ids.append(item.input_ids + [self.pad_token_id] * pad)
            labels.append(item.labels + [-100] * pad)
            attention.append([1] * len(item) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
