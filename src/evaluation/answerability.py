"""Answerability confusion matrix and Useful Answer Rate.

The hallucination rate alone is a trap: a model that refuses every question
scores a perfect 0.0 and is worthless. These metrics measure both directions of
the decision at once.

                            PREDICTION
                       answer        refuse
    ANSWER EXISTS        TP            FN     <- FN is over-refusal
    NO ANSWER            FP            TN     <- FP is hallucination

`answerable` in {full, partial} means an answer exists; {none} means it does
not. A *partial* question counts as answerable: the documented half still has
to be delivered, with the gap flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

# An answer is "useful" only if it is substantive, grounded and on target.
USEFUL_KEY_FACT_FLOOR = 0.5
USEFUL_F1_FLOOR = 0.2


@dataclass
class ConfusionMatrix:
    true_positive: int = 0    # answerable, answered
    false_negative: int = 0   # answerable, refused    -> over-refusal
    false_positive: int = 0   # unanswerable, answered -> hallucination
    true_negative: int = 0    # unanswerable, refused

    @property
    def answerable(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def unanswerable(self) -> int:
        return self.false_positive + self.true_negative

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    def as_dict(self) -> dict:
        tp, fn, fp, tn = (self.true_positive, self.false_negative,
                          self.false_positive, self.true_negative)
        return {
            "counts": {"true_positive": tp, "false_negative": fn,
                       "false_positive": fp, "true_negative": tn},
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            # Of everything it chose to answer, how much should it have answered
            "answer_precision": self._ratio(tp, tp + fp),
            # Of everything it could have answered, how much did it
            "answer_recall": self._ratio(tp, tp + fn),
            # Of everything it refused, how much deserved a refusal
            "refusal_precision": self._ratio(tn, tn + fn),
            # Of everything that deserved a refusal, how much did it refuse
            "refusal_recall": self._ratio(tn, tn + fp),
            "hallucination_rate": self._ratio(fp, fp + tn),
            "incorrect_refusal_rate": self._ratio(fn, tp + fn),
            "correct_refusal_rate": self._ratio(tn, fp + tn),
        }


def is_answerable(row: dict) -> bool:
    return str(row.get("answerable")) in ("full", "partial")


def predicted_refusal(row: dict) -> bool:
    """Did the model decline to answer, rather than answer with a caveat?

    `evaluate_example` computes this once as `wholesale_refusal` -- a refusal
    marker plus no usable citation plus no key facts delivered. Re-deriving it
    here from the published metrics got it wrong, so it is read, not inferred.
    """
    return bool(row.get("wholesale_refusal"))


def is_useful(row: dict) -> bool:
    """Did the model deliver a usable answer where one was possible?

    Requires all of: an answer was attempted, nothing was fabricated, at least
    one citation resolved, and the content is on target -- measured by required
    key facts when the example declares them, and by overlap with the reference
    otherwise.
    """
    if not is_answerable(row) or predicted_refusal(row):
        return False
    if row.get("hallucinated") or row.get("empty"):
        return False
    if not row.get("citation_precision"):
        return False
    key_facts = row.get("key_fact_recall")
    if key_facts is not None and row.get("_has_key_facts"):
        return key_facts >= USEFUL_KEY_FACT_FLOOR
    return (row.get("token_f1") or 0.0) >= USEFUL_F1_FLOOR


def confusion_matrix(rows: list[dict]) -> ConfusionMatrix:
    matrix = ConfusionMatrix()
    for row in rows:
        answerable, refused = is_answerable(row), predicted_refusal(row)
        if answerable and not refused:
            matrix.true_positive += 1
        elif answerable and refused:
            matrix.false_negative += 1
        elif not answerable and not refused:
            matrix.false_positive += 1
        else:
            matrix.true_negative += 1
    return matrix


def useful_answer_rate(rows: list[dict]) -> float | None:
    """Share of answerable questions that got a usable answer.

    Denominator is answerable questions only -- refusing an unanswerable
    question is correct and must not inflate this number.
    """
    answerable = [r for r in rows if is_answerable(r)]
    if not answerable:
        return None
    return round(sum(1 for r in answerable if is_useful(r)) / len(answerable), 4)


def answerability_report(rows: list[dict]) -> dict:
    matrix = confusion_matrix(rows)
    report = matrix.as_dict()
    report["useful_answer_rate"] = useful_answer_rate(rows)
    return report
