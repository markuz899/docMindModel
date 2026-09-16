import pytest

from src.config import ChunkingConfig, GenerationPipelineConfig, resolve_path
from src.dataset.schema import ContextChunk
from src.generation.markdown import load_corpus, split_markdown
from src.generation.providers import get_provider
from src.generation.teacher import (
    build_context_bundles,
    build_teacher_prompt,
    item_to_example,
    parse_teacher_output,
)
from src.prompt import SYSTEM_PROMPT, build_messages, build_user_message, citation

MD = """# Service

Overview paragraph long enough to survive the minimum chunk length that the
chunker enforces on every section it emits.

## GET /me

Returns the current user through UserService and the profile repository, with
enough text here to clear the minimum length.

```python
# this hash is code, not a heading
print(1)
```

## Errors

Returns 401 when the token is missing; the gateway maps it to a generic payload
and logs the request id for correlation.
"""


def test_chunker_splits_on_headings_and_ignores_code_fences():
    chunks = split_markdown(MD, "01-x.md")
    assert [c.heading for c in chunks] == ["Service", "GET /me", "Errors"]
    assert "not a heading" in chunks[1].content


def test_chunker_drops_sections_below_the_minimum():
    chunks = split_markdown("# A\n\ntiny\n", "x.md", ChunkingConfig(min_chunk_chars=80))
    assert chunks == []


def test_corpus_is_grouped_by_project():
    corpus = load_corpus("data/raw/projects")
    if not corpus:
        pytest.skip("no documentation corpus present")
    assert len(corpus) >= 2
    assert all(chunks for chunks in corpus.values())


def test_citation_format_is_stable():
    assert citation("04-api-reference.md", "GET /me") == "[04-api-reference.md — GET /me]"


def test_prompt_contract(chunks):
    payload = [c.model_dump() for c in chunks]
    user = build_user_message("Come funziona GET /me?", payload)
    # the citation token the model must copy appears verbatim above each block
    assert "[04-api-reference.md — GET /me]" in user
    assert user.index("## Project documentation") < user.index("## Question")
    messages = build_messages("Q?", payload, answer="A")
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_teacher_output_parsing_tolerates_prose_and_fences():
    raw = 'Sure!\n```json\n[{"question": "q?", "answer": "a"}]\n```\nHope that helps.'
    assert parse_teacher_output(raw) == [{"question": "q?", "answer": "a"}]
    assert parse_teacher_output("not json at all") == []
    assert parse_teacher_output('{"examples": [{"question": "q?"}]}') == [{"question": "q?"}]


def test_context_bundles_mix_relevant_and_distractor_sources():
    corpus = {
        "p1": [ContextChunk(source="a.md", heading="GET /me", content="x " * 60),
               ContextChunk(source="a.md", heading="Errors", content="y " * 60)],
        "p2": [ContextChunk(source="b.md", heading="Stripe", content="z " * 60)],
    }
    cfg = GenerationPipelineConfig(max_contexts=20, distractor_ratio=1.0, seed=1)
    bundles = build_context_bundles(corpus, cfg)
    assert bundles
    noisy = [b for b in bundles if len(b.chunks) > len(b.relevant)]
    assert noisy, "distractor_ratio=1.0 should always add irrelevant sources"
    assert any(c.relevant is False for c in noisy[0].chunks)


def test_mock_provider_round_trip_produces_valid_examples():
    corpus = load_corpus("data/raw/projects")
    if not corpus:
        pytest.skip("no documentation corpus present")
    cfg = GenerationPipelineConfig(max_contexts=3, seed=5)
    bundle = build_context_bundles(corpus, cfg)[0]
    provider = get_provider("mock")
    raw = provider.complete("system", build_teacher_prompt(bundle, cfg))
    items = parse_teacher_output(raw)
    assert len(items) == 2
    examples = [item_to_example(i, bundle, "teacher:mock") for i in items]
    assert all(e is not None for e in examples)
    assert {e.answerable for e in examples} == {"full", "none"}


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        get_provider("definitely-not-a-provider")
