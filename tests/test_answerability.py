"""Answerability confusion matrix, Useful Answer Rate, deduplication."""

import pytest

from src.config import DedupConfig, resolve_path
from src.dataset.dedup import deduplicate, normalize_question
from src.dataset.io import read_jsonl
from src.dataset.schema import ContextChunk, Example
from src.evaluation.answerability import (
    answerability_report,
    confusion_matrix,
    is_useful,
    useful_answer_rate,
)
from src.evaluation.metrics import aggregate, evaluate_example


@pytest.fixture
def suite(chunks):
    """One answerable and one unanswerable example over the same context."""
    answerable = Example(
        id="a", question="Come funziona GET /me?", context=chunks,
        answer="È gestita da `UserService` [04-api-reference.md — GET /me].",
        answerable="full", project="p", must_include=["UserService"],
        relevant_sources=["04-api-reference.md#GET /me"],
    )
    unanswerable = Example(
        id="u", question="Qual è la latenza p99 di GET /me?", context=chunks,
        answer="La documentazione fornita non riporta la latenza p99.",
        answerable="none", project="p",
    )
    return [answerable, unanswerable]


def _rows(suite, prediction_for):
    return [evaluate_example(e, prediction_for(e)) for e in suite]


# --- confusion matrix -----------------------------------------------------

def test_perfect_model_fills_only_the_diagonal(suite):
    matrix = confusion_matrix(_rows(suite, lambda e: e.answer))
    assert matrix.true_positive == 1 and matrix.true_negative == 1
    assert matrix.false_positive == 0 and matrix.false_negative == 0


def test_a_model_that_always_answers_hallucinates(suite):
    rows = _rows(suite, lambda e: "È gestita da `UserService` [04-api-reference.md — GET /me].")
    report = answerability_report(rows)
    assert report["counts"]["false_positive"] == 1
    assert report["hallucination_rate"] == 1.0
    assert report["incorrect_refusal_rate"] == 0.0


def test_a_model_that_always_refuses_scores_zero_hallucination_but_no_use(suite):
    """The whole reason Useful Answer Rate exists."""
    rows = _rows(suite, lambda e: "La documentazione fornita non contiene la risposta.")
    report = answerability_report(rows)
    assert report["hallucination_rate"] == 0.0, "refusing everything hides hallucination"
    assert report["incorrect_refusal_rate"] == 1.0
    assert report["useful_answer_rate"] == 0.0, "and Useful Answer Rate exposes it"


def test_precision_and_recall_are_reported_for_both_decisions(suite):
    report = answerability_report(_rows(suite, lambda e: e.answer))
    for key in ("answer_precision", "answer_recall",
                "refusal_precision", "refusal_recall"):
        assert report[key] == 1.0


# --- useful answer rate ---------------------------------------------------

def test_useful_requires_grounding_and_a_citation(suite):
    answerable = suite[0]
    uncited = evaluate_example(answerable, "È gestita da UserService.")
    assert not is_useful(uncited), "an uncited answer is not usable for DocMind"

    fabricated = evaluate_example(answerable, "Vedi [99-ghost.md — Nowhere].")
    assert not is_useful(fabricated)

    good = evaluate_example(answerable, answerable.answer)
    assert is_useful(good)


def test_unanswerable_questions_are_excluded_from_the_denominator(suite):
    rows = _rows(suite, lambda e: e.answer)
    assert useful_answer_rate(rows) == 1.0  # 1 of 1 answerable, not 1 of 2


def test_useful_answer_rate_is_none_without_answerable_examples():
    assert useful_answer_rate([]) is None


def test_aggregate_exposes_the_answerability_block(suite):
    summary = aggregate(_rows(suite, lambda e: e.answer))
    assert "answerability" in summary and "useful_answer_rate" in summary


def test_oracle_is_useful_on_every_shipped_suite():
    suites = ["data/test/test.jsonl", "benchmarks/bug_investigation.jsonl",
              "benchmarks/retrieval_noise.jsonl"]
    missing = [s for s in suites if not resolve_path(s).exists()]
    if missing:
        pytest.skip(f"not built: {missing}")
    for path in suites:
        rows = [evaluate_example(e, e.answer) for e in read_jsonl(path)]
        assert useful_answer_rate(rows) >= 0.95, path


def test_noise_benchmark_penalises_citing_a_distractor():
    path = resolve_path("benchmarks/retrieval_noise.jsonl")
    if not path.exists():
        pytest.skip("benchmarks not built")
    example = next(e for e in read_jsonl(path)
                   if any(c.relevant is False for c in e.context))
    distractor = next(c for c in example.context if c.relevant is False)
    row = evaluate_example(example, f"Vedi [{distractor.source} — {distractor.heading}].")
    assert row["distractor_citations"] == 1
    assert row["relevant_source_usage"] == 0.0


# --- deduplication --------------------------------------------------------

def _q(i, question, category="how_it_works", ctx=None):
    ctx = ctx or [ContextChunk(source="a.md", heading="GET /me",
                               content="Returns the user through UserService." * 4)]
    return Example(id=f"e{i}", question=question, context=ctx,
                   answer="Vedi [a.md — GET /me].", answerable="full",
                   project="p", category=category)


def test_normalisation_ignores_phrasing_and_punctuation():
    assert normalize_question("Come funziona la /me?") == normalize_question("/me come FUNZIONA")


def test_exact_duplicate_questions_are_dropped():
    kept, dropped = deduplicate([_q(0, "Come funziona /me?"), _q(1, "Come funziona /me?")])
    assert len(kept) == 1 and dropped[0]["reason"] == "duplicate question"


def test_rewordings_of_the_same_subject_are_dropped():
    """The failure mode this exists for: five ways to ask one question."""
    questions = ["Come funziona /me?", "Puoi spiegare /me?", "Cosa fa la /me?",
                 "Come opera /me?"]
    kept, dropped = deduplicate([_q(i, q) for i, q in enumerate(questions)])
    assert len(kept) == 1
    assert all("same subject" in d["reason"] or "duplicate" in d["reason"] for d in dropped)


def test_a_different_angle_on_the_same_symbol_is_kept():
    kept, _ = deduplicate([
        _q(0, "Come funziona /me?", "how_it_works"),
        _q(1, "Quali errori restituisce /me?", "error_handling"),
    ])
    assert len(kept) == 2, "different category means a genuinely different question"


def test_one_context_cannot_dominate_the_dataset():
    examples = [_q(i, f"Domanda distinta numero {i} su componente{i}?") for i in range(10)]
    kept, dropped = deduplicate(examples, DedupConfig(per_context_limit=3))
    assert len(kept) == 3
    assert any("per_context_limit" in d["reason"] for d in dropped)


def test_dedup_can_be_disabled():
    examples = [_q(0, "Come funziona /me?"), _q(1, "Come funziona /me?")]
    kept, dropped = deduplicate(examples, DedupConfig(enabled=False))
    assert len(kept) == 2 and not dropped
