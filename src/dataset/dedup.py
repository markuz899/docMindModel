"""Question-level deduplication.

A teacher asked for five examples from one context will happily produce
"Come funziona /me?", "Puoi spiegare /me?", "Cosa fa /me?" -- three rows that
look like data and teach nothing. Left alone they also inflate every metric,
because the test set ends up holding rewordings of what the model trained on.

Two passes: drop near-identical questions that share a context, then cap how
many questions any single context may contribute at all.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.config import DedupConfig
from src.dataset.schema import Example
from src.text import STOPWORDS, technical_tokens, tokenize

_PUNCT = re.compile(r"[^\w\s/.:_-]+")


def normalize_question(question: str) -> str:
    """Lowercase, strip punctuation and filler, so phrasing stops mattering."""
    text = _PUNCT.sub(" ", (question or "").lower())
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]
    return " ".join(sorted(set(tokens)))


def _signature(example: Example) -> tuple[str, ...]:
    return tuple(sorted(f"{s}#{h}" for s, h in example.context_keys()))


def _subject(example: Example) -> frozenset[str]:
    """What the question is *about*: its identifiers, routes, tables, keys."""
    return frozenset(t.lower() for t in technical_tokens(example.question))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    examples: list[Example], cfg: DedupConfig | None = None
) -> tuple[list[Example], list[dict]]:
    """-> (kept, dropped) where each dropped row says what it duplicated."""
    cfg = cfg or DedupConfig()
    if not cfg.enabled:
        return list(examples), []

    kept: list[Example] = []
    dropped: list[dict] = []
    seen_exact: dict[str, str] = {}
    by_context: dict[tuple[str, ...], list[tuple[set[str], Example]]] = defaultdict(list)
    # (context, technical subject, category, answerability) -> the example that
    # claimed it. Purely lexical similarity cannot tell "Come funziona /me?"
    # from "Puoi spiegare /me?" -- they share almost no tokens -- but two
    # generated questions about the same symbol, over the same sources, in the
    # same category and answerability are rewordings of each other.
    claimed_subjects: dict[tuple, str] = {}

    for example in examples:
        normalized = normalize_question(example.question)
        if not normalized:
            dropped.append({"id": example.id, "reason": "empty question after normalisation"})
            continue

        # 1. the same question twice, anywhere
        if normalized in seen_exact:
            dropped.append({"id": example.id, "reason": "duplicate question",
                            "duplicate_of": seen_exact[normalized]})
            continue

        signature = _signature(example)
        tokens = set(normalized.split())
        bucket = by_context[signature]

        # 2. a reworded question over the very same sources
        near = next(
            (other for other_tokens, other in bucket
             if _jaccard(tokens, other_tokens) >= cfg.question_similarity),
            None,
        )
        if near is not None:
            dropped.append({"id": example.id, "reason": "near-duplicate question on same context",
                            "duplicate_of": near.id,
                            "similarity": round(max(
                                _jaccard(tokens, t) for t, _ in bucket), 3)})
            continue

        subject = _subject(example)
        subject_key = (signature, subject, str(example.category), str(example.answerable))
        if subject and subject_key in claimed_subjects:
            dropped.append({"id": example.id,
                            "reason": "same subject, sources, category and answerability",
                            "duplicate_of": claimed_subjects[subject_key],
                            "subject": sorted(subject)})
            continue

        # 3. one context may only contribute so much of the dataset
        if cfg.per_context_limit and len(bucket) >= cfg.per_context_limit:
            dropped.append({"id": example.id, "reason": "context already at per_context_limit",
                            "limit": cfg.per_context_limit})
            continue

        seen_exact[normalized] = example.id
        if subject:
            claimed_subjects[subject_key] = example.id
        bucket.append((tokens, example))
        kept.append(example)

    return kept, dropped
