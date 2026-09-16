import pytest

from src.config import resolve_path
from src.dataset.io import read_jsonl
from src.dataset.schema import Example
from src.evaluation.metrics import aggregate, evaluate_example


def test_reference_answer_scores_near_perfect(good_example):
    row = evaluate_example(good_example, good_example.answer)
    assert row["answer_correctness"] > 0.95
    assert row["groundedness"] == 1.0
    assert row["citation_f1"] == 1.0
    assert row["relevant_source_usage"] == 1.0
    assert row["hallucinated"] is False


def test_confident_fabrication_is_caught(good_example):
    row = evaluate_example(
        good_example, "È un NullPointerException in OrderService [99-ghost.md — Nowhere]."
    )
    assert row["hallucinated"] is True
    assert row["fabricated_citations"]
    assert row["answer_correctness"] < 0.2


def test_failing_to_refuse_counts_as_hallucination(chunks):
    unanswerable = Example(
        id="n", question="Qual è la latenza p99?", context=chunks,
        answer="La documentazione non riporta la latenza p99.", answerable="none",
        project="demo",
    )
    good = evaluate_example(unanswerable, unanswerable.answer)
    assert good["refusal_correct"] and not good["hallucinated"]

    bad = evaluate_example(unanswerable, "La latenza p99 è circa 120 ms.")
    assert not bad["refusal_correct"] and bad["hallucinated"]


def test_caveat_inside_a_real_answer_is_not_a_false_refusal(good_example):
    answer = good_example.answer + " Non documentato: le fonti non citano la cache."
    row = evaluate_example(good_example, answer)
    assert row["false_refusal"] is False


def test_wholesale_refusal_on_an_answerable_question_is_penalised(good_example):
    row = evaluate_example(good_example, "La documentazione fornita non contiene la risposta.")
    assert row["false_refusal"] is True
    assert row["refusal_correct"] is False


def test_distractor_citations_are_counted(chunks):
    noisy = list(chunks) + [
        chunks[0].model_copy(update={"source": "99-other.md", "heading": "Kubernetes",
                                     "relevant": False})
    ]
    ex = Example(id="d", question="Come funziona GET /me?", context=noisy,
                 answer="x [04-api-reference.md — GET /me].", answerable="full",
                 project="demo",
                 relevant_sources=["04-api-reference.md#GET /me"])
    clean = evaluate_example(ex, "Vedi [04-api-reference.md — GET /me].")
    assert clean["distractor_citations"] == 0
    dirty = evaluate_example(ex, "Vedi [99-other.md — Kubernetes].")
    assert dirty["distractor_citations"] == 1
    assert dirty["relevant_source_usage"] == 0.0


def test_conciseness_penalises_rambling(good_example):
    padded = good_example.answer + " parola" * 400
    assert evaluate_example(good_example, padded)["conciseness"] < 0.2


def test_aggregate_reports_the_headline_rates(good_example):
    rows = [evaluate_example(good_example, good_example.answer)]
    summary = aggregate(rows)
    assert summary["count"] == 1
    assert summary["hallucinated_rate"] == 0.0
    assert "false_refusal_rate_answerable" in summary


def test_oracle_scores_high_on_every_shipped_suite():
    suites = ["data/test/test.jsonl", "benchmarks/hallucination.jsonl",
              "benchmarks/bug_investigation.jsonl", "benchmarks/retrieval_noise.jsonl"]
    missing = [s for s in suites if not resolve_path(s).exists()]
    if missing:
        pytest.skip(f"not built yet: {missing}")
    for suite in suites:
        rows = [evaluate_example(e, e.answer) for e in read_jsonl(suite)]
        summary = aggregate(rows)
        assert summary["hallucinated_rate"] == 0.0, suite
        assert summary["refusal_correct_rate"] == 1.0, suite
        assert summary["answer_correctness"] > 0.95, suite


def test_every_shipped_refusal_template_is_detected_as_a_refusal():
    """A phrasing the detector misses is a refusal scored as a hallucination."""
    from src.dataset.negatives import REFUSAL_TEMPLATES_EN, REFUSAL_TEMPLATES_IT
    from src.text import looks_like_refusal

    missed = []
    for templates in (REFUSAL_TEMPLATES_IT, REFUSAL_TEMPLATES_EN):
        for template in templates:
            body = template.format(
                topic="the p99 latency", topic_cap="The p99 latency",
                subject="GET /me", headings='"A", "B" and "C"', need="the runbook",
            )
            if not looks_like_refusal(body):
                missed.append(body.splitlines()[0])
    assert not missed, missed
