"""Automatic quality gates.

Anything that reaches training has passed these. The expensive lesson this
encodes: a single fabricated citation in the training set teaches the model
that fabricating citations is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import QualityConfig
from src.dataset.schema import Answerability, Example
from src.text import (
    flags_contradiction,
    groundedness,
    technical_groundedness,
    looks_like_refusal,
    parse_citations,
    normalize,
    unsupported_identifiers,
)


@dataclass
class Report:
    example_id: str
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def check_example(example: Example, cfg: QualityConfig | None = None) -> Report:
    cfg = cfg or QualityConfig()
    report = Report(example_id=example.id)
    issues = report.issues

    question, answer = example.question.strip(), example.answer.strip()
    context_text = example.context_text()

    # --- emptiness / size -------------------------------------------------
    if len(question) < cfg.min_question_chars:
        issues.append(f"question too short ({len(question)} chars)")
    if len(question) > cfg.max_question_chars:
        issues.append(f"question too long ({len(question)} chars)")
    if len(answer) < cfg.min_answer_chars:
        issues.append(f"answer too short ({len(answer)} chars)")
    if len(answer) > cfg.max_answer_chars:
        issues.append(f"answer too long ({len(answer)} chars)")
    if context_text.strip() == "":
        issues.append("empty context")

    ratio = len(answer) / max(len(context_text), 1)
    report.metrics["answer_context_ratio"] = round(ratio, 3)
    # A faithful answer over a two-paragraph context is legitimately about as
    # long as its source; only long *and* disproportionate answers are rambling.
    if len(answer) > cfg.long_answer_min_chars and ratio > cfg.max_answer_context_ratio:
        issues.append(
            f"answer rambles: {len(answer)} chars, {ratio:.2f}x the context"
        )

    # --- citations must resolve to a supplied chunk -----------------------
    available = {(s.lower(), h.lower()) for s, h in example.context_keys()}
    available_sources = {s.lower() for s, _ in example.context_keys()}
    cited = parse_citations(answer)
    report.metrics["citation_count"] = len(cited)
    for source, heading in cited:
        if (source.lower(), heading.lower()) in available:
            continue
        if source.lower() in available_sources:
            issues.append(f"citation heading not in context: [{source} - {heading}]")
        else:
            issues.append(f"fabricated citation source: [{source} - {heading}]")

    answerability = example.answerable or Answerability.FULL
    is_refusal_case = answerability == Answerability.NONE

    if cfg.require_citation_when_answerable and not is_refusal_case and not cited:
        issues.append("answerable example without any citation")

    # --- grounding --------------------------------------------------------
    unsupported = unsupported_identifiers(answer, context_text, question)
    report.metrics["unsupported_identifiers"] = len(unsupported)
    # A refusal has to name the thing that is missing ("no SLA is documented"),
    # so it gets a small budget instead of the strict zero.
    budget = (
        cfg.max_unsupported_identifiers_refusal
        if is_refusal_case
        else cfg.max_unsupported_identifiers
    )
    if len(unsupported) > budget:
        issues.append("identifiers absent from context: " + ", ".join(sorted(unsupported)[:6]))

    report.metrics["groundedness"] = round(groundedness(answer, context_text), 3)
    tech = technical_groundedness(answer, context_text)
    report.metrics["technical_groundedness"] = round(tech, 3)
    # Prose overlap is language-sensitive (an Italian answer over English docs
    # shares little wording), so the gate is on technical tokens instead.
    if not is_refusal_case and tech < cfg.min_technical_groundedness:
        issues.append(f"technical groundedness {tech:.2f} below {cfg.min_technical_groundedness}")

    # --- behavioural contracts -------------------------------------------
    if is_refusal_case and not looks_like_refusal(answer):
        issues.append("answerable=none but the answer does not state the gap")
    is_contradiction = "contradiction" in example.tags
    if answerability == Answerability.PARTIAL and not is_contradiction and not looks_like_refusal(answer):
        issues.append("answerable=partial but the answer never flags the unknown part")
    if is_contradiction:
        if not flags_contradiction(answer):
            issues.append("contradiction example that never names the conflict")
        if len({s for s, _ in cited}) < 2 and len(cited) < 2:
            issues.append("contradiction example must cite both conflicting sources")

    # --- explicit negative facts -----------------------------------------
    haystack = normalize(answer)
    for forbidden in example.must_not_include:
        if normalize(forbidden) in haystack:
            issues.append(f"answer asserts ungrounded fact: {forbidden!r}")

    # --- gold sources must exist in the context ---------------------------
    for key in example.relevant_keys():
        if (key[0].lower(), key[1].lower()) not in available:
            issues.append(f"relevant_sources points outside the context: {key[0]}#{key[1]}")

    return report


def filter_examples(
    examples: list[Example], cfg: QualityConfig | None = None
) -> tuple[list[Example], list[Report]]:
    """Split a batch into (accepted, rejection reports). Duplicate ids are rejected."""
    cfg = cfg or QualityConfig()
    kept, rejected, seen_ids = [], [], set()
    for ex in examples:
        report = check_example(ex, cfg)
        if ex.id in seen_ids:
            report.issues.append("duplicate id")
        seen_ids.add(ex.id)
        (kept if report.ok else rejected).append(ex if report.ok else report)
    return kept, rejected
