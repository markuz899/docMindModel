"""Training-path tests. Skipped when the tiny local model has not been built."""

import pytest

from src.config import load_model_config, resolve_path

TINY = "artifacts/tiny-model"
pytestmark = pytest.mark.skipif(
    not (resolve_path(TINY) / "config.json").exists(),
    reason="run scripts/make_tiny_model.py first",
)


@pytest.fixture(scope="module")
def tokenizer():
    from src.runtime import load_tokenizer

    return load_tokenizer(load_model_config("configs/model.smoke.yaml"))


def test_only_the_answer_contributes_to_the_loss(tokenizer, good_example):
    from src.training.data import encode_example

    encoded = encode_example(good_example, tokenizer, max_seq_length=2048)
    assert len(encoded.input_ids) == len(encoded.labels)
    supervised = [t for t in encoded.labels if t != -100]
    assert supervised, "nothing is being trained on"
    assert len(supervised) < len(encoded.input_ids), "the prompt is not masked"
    decoded = tokenizer.decode(supervised)
    assert good_example.answer[:40] in decoded


def test_long_context_is_truncated_but_the_answer_survives(tokenizer, good_example):
    from src.training.data import encode_example

    encoded = encode_example(good_example, tokenizer, max_seq_length=64)
    assert len(encoded.input_ids) <= 64
    assert any(t != -100 for t in encoded.labels)


def test_collator_pads_and_masks(tokenizer, good_example):
    from src.training.data import CompletionCollator, encode_dataset

    encoded, stats = encode_dataset(
        [good_example, good_example.model_copy(update={"id": "t-002", "answer": "Breve."})],
        tokenizer,
        max_seq_length=2048,
    )
    batch = CompletionCollator(tokenizer.pad_token_id)(encoded)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # padded positions are excluded from both attention and loss
    padded = batch["attention_mask"] == 0
    assert (batch["labels"][padded] == -100).all()
    assert stats["trainable_tokens"] > 0


def test_non_prefix_stable_chat_template_fails_loudly(tokenizer, good_example, monkeypatch):
    from src.training.data import encode_example

    def broken(messages, **kwargs):
        return "PREFIX-" if kwargs.get("add_generation_prompt") else "SOMETHING-ELSE"

    monkeypatch.setattr(tokenizer, "apply_chat_template", broken)
    with pytest.raises(ValueError, match="prefix-stable"):
        encode_example(good_example, tokenizer, max_seq_length=128)


def test_engine_generates_and_renders_the_contract(good_example):
    from src.inference.engine import DocMindEngine

    engine = DocMindEngine.load("configs/model.smoke.yaml")
    payload = [c.model_dump() for c in good_example.context]
    prompt = engine.render_prompt(good_example.question, payload)
    assert "[04-api-reference.md — GET /me]" in prompt
    assert prompt.rstrip().endswith("assistant") or "assistant" in prompt[-40:]
    out = engine.generate_batch([(good_example.question, payload)], max_new_tokens=4)
    assert isinstance(out, list) and len(out) == 1


def test_overlong_examples_are_dropped_not_truncated(tokenizer, good_example):
    """Truncating the prompt can delete the section the answer cites."""
    from src.training.data import encode_dataset

    encoded, stats = encode_dataset([good_example], tokenizer, max_seq_length=64)
    assert encoded == [] and stats["skipped_overlong"] == 1 and stats["truncated"] == 0

    encoded, stats = encode_dataset(
        [good_example], tokenizer, max_seq_length=64, skip_overlong=False
    )
    assert len(encoded) == 1 and stats["truncated"] == 1
