"""The seven evaluators.

All lexical and deterministic: no judge model, no network, same numbers on a
laptop and in CI. They measure behaviour, not style:

  1. answer_correctness      overlap with the reference + required key facts
  2. groundedness            are the asserted facts in the context at all
  3. citation_correctness    do the citations resolve, and to the right sections
  4. refusal_correctness     refuse when it should, answer when it should
  5. hallucination_rate      the headline metric
  6. relevant_source_usage   uses the relevant sources, ignores the distractors
  7. conciseness             developer-useful length, not an essay
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from src.dataset.schema import Answerability, Example
from src.evaluation.answerability import answerability_report
from src.text import (
    keyword_recall,
    leads_with_refusal,
    looks_like_refusal,
    normalize,
    parse_citations,
    technical_groundedness,
    token_f1,
    unsupported_identifiers,
    groundedness as lexical_groundedness,
)

# A refusal names the missing topic, which may not be in the context.
REFUSAL_IDENTIFIER_BUDGET = 3
CONCISENESS_TARGET_RATIO = 1.5


def _cited_keys(prediction: str) -> list[tuple[str, str]]:
    return [(s.strip().lower(), h.strip().lower()) for s, h in parse_citations(prediction)]


def evaluate_example(example: Example, prediction: str) -> dict:
    prediction = (prediction or "").strip()
    context_text = example.context_text()
    answerability = example.answerable or Answerability.FULL
    expects_refusal = answerability in (Answerability.NONE, Answerability.PARTIAL)

    # 1. answer correctness ------------------------------------------------
    f1 = token_f1(prediction, example.answer)
    # `facts` is what a real teacher emits; `must_include` is the hand-written
    # equivalent. Either one makes key-fact recall meaningful.
    key_facts = example.must_include or example.facts
    kw = keyword_recall(prediction, key_facts)
    correctness = 0.5 * f1 + 0.5 * kw if key_facts else f1

    refused = looks_like_refusal(prediction)

    # 2. groundedness ------------------------------------------------------
    unsupported = unsupported_identifiers(prediction, context_text, example.question)
    grounded = technical_groundedness(prediction, context_text)

    # 3. citation correctness ---------------------------------------------
    available = {(s.lower(), h.lower()) for s, h in example.context_keys()}
    available_sources = {s.lower() for s, _ in example.context_keys()}
    cited = _cited_keys(prediction)
    fabricated = [c for c in cited if c[0] not in available_sources]
    misattributed = [c for c in cited if c[0] in available_sources and c not in available]
    resolvable = [c for c in cited if c in available]
    # Citing the same section twice is style, not an extra citation.
    unique_cited, unique_resolvable = set(cited), set(resolvable)
    # No citations is only acceptable when the answer is a refusal; an
    # uncited substantive answer scores zero, not one.
    citation_precision = (
        len(unique_resolvable) / len(unique_cited) if unique_cited
        else (1.0 if refused else 0.0)
    )

    gold = {(s.lower(), h.lower()) for s, h in example.relevant_keys()}
    gold_cited = {c for c in unique_resolvable if c in gold}
    citation_recall = len(gold_cited) / len(gold) if gold else 1.0
    citation_f1 = (
        0.0
        if citation_precision + citation_recall == 0
        else 2 * citation_precision * citation_recall / (citation_precision + citation_recall)
    )

    # 4. refusal correctness ----------------------------------------------
    # Flagging a gap inside an otherwise substantive answer is correct
    # behaviour, so only a *wholesale* refusal (no usable citation, no key
    # facts) counts as refusing to answer an answerable question.
    # `keyword_recall` returns 1.0 for an empty key-fact list, so that clause
    # must only apply when the example actually declares key facts -- otherwise
    # a pure refusal never registers as one.
    has_key_facts = bool(key_facts)
    delivered_key_facts = kw >= 0.5 if has_key_facts else False
    # A refusal that also cites what *is* documented is still a refusal, so the
    # test is where the decline appears, not whether citations exist.
    wholesale_refusal = (
        refused
        and not delivered_key_facts
        and (leads_with_refusal(prediction) or not resolvable)
    )
    if answerability == Answerability.FULL:
        refusal_correct = not wholesale_refusal
        false_refusal = wholesale_refusal
    else:
        refusal_correct = refused
        false_refusal = False

    # 5. hallucination -----------------------------------------------------
    forbidden_hits = [
        t for t in example.must_not_include if normalize(t) in normalize(prediction)
    ]
    budget = REFUSAL_IDENTIFIER_BUDGET if answerability == Answerability.NONE else 0
    hallucinated = bool(
        fabricated
        or misattributed
        or forbidden_hits
        or len(unsupported) > budget
        or (answerability == Answerability.NONE and not refused)
    )

    # 6. relevant source usage --------------------------------------------
    distractor_keys = {
        (c.source.lower(), c.heading.lower()) for c in example.context if c.relevant is False
    }
    distractor_citations = [c for c in unique_cited if c in distractor_keys]
    relevant_usage = (
        len(gold_cited) / len(unique_resolvable) if unique_resolvable
        else (1.0 if refused else 0.0)
    )

    # 7. conciseness -------------------------------------------------------
    pred_words = len(prediction.split())
    ref_words = max(len(example.answer.split()), 1)
    ratio = pred_words / ref_words
    conciseness = 1.0 if ratio <= CONCISENESS_TARGET_RATIO else CONCISENESS_TARGET_RATIO / ratio

    return {
        "id": example.id,
        "project": example.project,
        "category": example.category,
        "answerable": str(answerability),
        "tags": example.tags,
        "empty": not prediction,
        "answer_correctness": round(correctness, 4),
        "token_f1": round(f1, 4),
        "key_fact_recall": round(kw, 4),
        "_has_key_facts": has_key_facts,
        "groundedness": round(grounded, 4),
        "lexical_groundedness": round(lexical_groundedness(prediction, context_text), 4),
        "unsupported_identifiers": sorted(unsupported),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "citation_f1": round(citation_f1, 4),
        "fabricated_citations": [f"{s} — {h}" for s, h in fabricated],
        "misattributed_citations": [f"{s} — {h}" for s, h in misattributed],
        "refused": bool(refused),
        "wholesale_refusal": bool(wholesale_refusal),
        "refusal_correct": bool(refusal_correct),
        "false_refusal": bool(false_refusal),
        "hallucinated": hallucinated,
        "forbidden_facts": forbidden_hits,
        "relevant_source_usage": round(relevant_usage, 4),
        "distractor_citations": len(distractor_citations),
        "conciseness": round(conciseness, 4),
        "length_ratio": round(ratio, 3),
        "words": pred_words,
    }


_MEAN_KEYS = (
    "answer_correctness", "token_f1", "key_fact_recall", "groundedness",
    "lexical_groundedness", "citation_precision", "citation_recall", "citation_f1",
    "relevant_source_usage", "conciseness", "length_ratio",
)
_RATE_KEYS = ("refusal_correct", "false_refusal", "hallucinated", "empty")


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    out: dict[str, object] = {"count": len(rows)}
    for key in _MEAN_KEYS:
        out[key] = round(mean(r[key] for r in rows), 4)
    for key in _RATE_KEYS:
        out[f"{key}_rate"] = round(mean(1.0 if r[key] else 0.0 for r in rows), 4)
    out["distractor_citations_per_answer"] = round(
        mean(r["distractor_citations"] for r in rows), 4
    )
    out["answerability"] = answerability_report(rows)
    out["useful_answer_rate"] = out["answerability"]["useful_answer_rate"]
    unanswerable = [r for r in rows if r["answerable"] == "none"]
    if unanswerable:
        out["hallucination_rate_unanswerable"] = round(
            mean(1.0 if r["hallucinated"] else 0.0 for r in unanswerable), 4
        )
        out["refusal_rate_unanswerable"] = round(
            mean(1.0 if r["refusal_correct"] else 0.0 for r in unanswerable), 4
        )
    answerable = [r for r in rows if r["answerable"] == "full"]
    if answerable:
        out["false_refusal_rate_answerable"] = round(
            mean(1.0 if r["false_refusal"] else 0.0 for r in answerable), 4
        )
    return out


def aggregate_by(rows: list[dict], field: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field))].append(row)
    return {key: aggregate(items) for key, items in sorted(buckets.items())}
