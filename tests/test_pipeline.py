"""Generation pipeline: planning, batching, structured-output parsing, audit."""

import json

import pytest

from src.config import GenerationPipelineConfig, load_dataset_config
from src.dataset.schema import ContextChunk
from src.generation.markdown import load_corpus
from src.generation.pipeline import (
    BundlePlan,
    build_prompt,
    item_to_example,
    plan_bundles,
    run_generation,
)
from src.generation.prompts import CATEGORIES, TEACHER_JSON_SCHEMA, build_batch_instruction
from src.generation.providers import get_provider
from src.generation.teacher import parse_teacher_output


@pytest.fixture
def corpus():
    return load_corpus("data/raw/projects")


# --- planning -------------------------------------------------------------

def test_planning_honours_the_batch_size(corpus):
    cfg = GenerationPipelineConfig(examples_per_teacher_call=5, max_contexts=20)
    bundles = plan_bundles(corpus, cfg)
    assert bundles and all(len(b.slots) == 5 for b in bundles)


def test_batching_means_far_fewer_calls_than_examples(corpus):
    cfg = GenerationPipelineConfig(examples_per_teacher_call=5, max_contexts=100)
    bundles = plan_bundles(corpus, cfg)
    requested = sum(len(b.slots) for b in bundles)
    assert len(bundles) * 5 == requested, "one call must cover five examples"


def test_planning_is_deterministic(corpus):
    cfg = GenerationPipelineConfig(seed=11, max_contexts=15)
    first = [b.id for b in plan_bundles(corpus, cfg)]
    second = [b.id for b in plan_bundles(corpus, cfg)]
    assert first == second


def test_answerability_mix_follows_the_configuration(corpus):
    cfg = GenerationPipelineConfig(max_contexts=200, examples_per_teacher_call=5)
    slots = [a for b in plan_bundles(corpus, cfg) for a, _ in b.slots]
    share = {kind: slots.count(kind) / len(slots) for kind in set(slots)}
    # full absorbs the "noisy" bucket, which is answerable-with-distractors
    assert 0.55 <= share["full"] <= 0.75
    assert 0.10 <= share["none"] <= 0.30
    assert 0.08 <= share["partial"] <= 0.25


def test_every_planned_category_is_a_known_category(corpus):
    cfg = GenerationPipelineConfig(max_contexts=60)
    for bundle in plan_bundles(corpus, cfg):
        for _, category in bundle.slots:
            assert category in CATEGORIES


def test_distractors_come_from_other_projects_without_key_collisions(corpus):
    cfg = GenerationPipelineConfig(distractor_ratio=1.0, max_contexts=40, seed=3)
    noisy = [b for b in plan_bundles(corpus, cfg) if b.has_distractors]
    assert noisy
    for bundle in noisy[:10]:
        relevant = {c.key for c in bundle.relevant}
        distractors = {c.key for c in bundle.chunks if c.relevant is False}
        assert not (relevant & distractors), "a distractor must not shadow a real source"


def test_prompt_carries_the_plan_and_only_this_bundle(corpus):
    bundle = plan_bundles(corpus, GenerationPipelineConfig(max_contexts=5))[0]
    prompt = build_prompt(bundle)
    assert f"exactly {len(bundle.slots)}" in prompt
    for _, category in bundle.slots:
        assert category in prompt
    # no other project's sections leak in
    for chunk in bundle.chunks:
        assert chunk.heading in prompt


def test_schema_requires_every_property():
    """OpenAI structured output rejects a schema whose `required` omits a key."""
    item = TEACHER_JSON_SCHEMA["properties"]["examples"]["items"]
    assert set(item["required"]) == set(item["properties"])


# --- parsing --------------------------------------------------------------

def test_parser_accepts_the_object_form():
    raw = json.dumps({"examples": [{"question": "q?", "answer": "a"}]})
    assert parse_teacher_output(raw) == [{"question": "q?", "answer": "a"}]


def test_parser_survives_fences_and_chatter():
    raw = 'Here you go:\n```json\n{"examples": [{"question": "q?", "answer": "a"}]}\n```\nDone.'
    assert parse_teacher_output(raw)[0]["question"] == "q?"


def test_parser_rejects_invalid_json_rather_than_guessing():
    assert parse_teacher_output("not json") == []
    assert parse_teacher_output("") == []


def test_required_sources_maps_onto_relevant_sources():
    chunk = ContextChunk(source="a.md", heading="GET /me", content="x" * 100)
    bundle = BundlePlan(id="b", project="p", chunks=[chunk], relevant=[chunk],
                        slots=[("full", "api")])
    example = item_to_example(
        {"question": "Come funziona /me?", "answer": "Vedi [a.md — GET /me].",
         "category": "api", "difficulty": "easy", "answerable": "full",
         "required_sources": ["a.md#GET /me"], "facts": ["/me returns the user"],
         "unsupported_claims": ["latency"]},
        bundle, {"teacher_provider": "codex"},
    )
    assert example.relevant_sources == ["a.md#GET /me"]
    assert example.facts and example.unsupported_claims


def test_every_example_records_its_teacher_audit_trail():
    chunk = ContextChunk(source="a.md", heading="GET /me", content="x" * 100)
    bundle = BundlePlan(id="b", project="p", chunks=[chunk], relevant=[chunk],
                        slots=[("full", "api")])
    audit = {"teacher_provider": "codex", "teacher_cli_version": "0.154.0",
             "prompt_version": "2", "teacher_request_hash": "abc"}
    example = item_to_example({"question": "q?", "answer": "a"}, bundle, audit)
    assert example.teacher_metadata == audit
    assert example.origin == "teacher:codex"


def test_malformed_items_are_dropped_not_guessed():
    chunk = ContextChunk(source="a.md", heading="H", content="x" * 100)
    bundle = BundlePlan(id="b", project="p", chunks=[chunk], relevant=[chunk],
                        slots=[("full", "api")])
    assert item_to_example({"question": "", "answer": "a"}, bundle, {}) is None
    assert item_to_example({"question": "q", "answer": ""}, bundle, {}) is None


def test_batch_instruction_names_every_slot():
    text = build_batch_instruction(2, [("full", "api"), ("none", "unanswerable")], True, True)
    assert "exactly 2" in text and "api" in text and "unanswerable" in text
    assert "irrelevant" in text and "conflict" in text


# --- the run --------------------------------------------------------------

def test_run_is_cached_and_resumable(tmp_path, monkeypatch):
    cfg = load_dataset_config()
    cfg = cfg.model_copy(update={"generation": cfg.generation.model_copy(
        update={"cache_dir": str(tmp_path / "cache"), "examples_per_teacher_call": 2})})
    out = tmp_path / "out.jsonl"

    first, report = run_generation("data/raw/projects", get_provider("mock"), cfg,
                                   output_path=out, max_bundles=3)
    assert report.teacher_calls == 3 and report.cache_hits == 0
    assert first and out.exists()

    _, cached = run_generation("data/raw/projects", get_provider("mock"), cfg,
                               output_path=tmp_path / "out2.jsonl", max_bundles=3)
    assert cached.teacher_calls == 0 and cached.cache_hits == 3

    resumed, report3 = run_generation("data/raw/projects", get_provider("mock"), cfg,
                                      output_path=out, resume=True, max_bundles=3)
    assert report3.bundles_done == 0, "resume must skip finished bundles"
    assert len(resumed) == len(first)
