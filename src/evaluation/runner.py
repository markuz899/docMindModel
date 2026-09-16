"""Run a model over an evaluation suite and score every answer."""

from __future__ import annotations

import time
from pathlib import Path

from src.dataset.io import read_jsonl
from src.dataset.schema import Example
from src.evaluation.metrics import aggregate, aggregate_by, evaluate_example
from src.inference.engine import DocMindEngine

SUITES: dict[str, str] = {
    "test": "data/test/test.jsonl",
    "validation": "data/validation/validation.jsonl",
    "hallucination": "benchmarks/hallucination.jsonl",
    "retrieval_noise": "benchmarks/retrieval_noise.jsonl",
    "bug_investigation": "benchmarks/bug_investigation.jsonl",
}


def load_suite(name: str, limit: int | None = None) -> list[Example]:
    path = SUITES.get(name, name)
    examples = read_jsonl(path)
    return examples[:limit] if limit else examples


def run_suite(
    engine: DocMindEngine,
    examples: list[Example],
    batch_size: int = 1,
    max_new_tokens: int | None = None,
    progress_every: int = 10,
    name: str = "suite",
) -> dict:
    items = [(ex.question, [c.model_dump() for c in ex.context]) for ex in examples]
    started = time.time()
    predictions: list[str] = []
    for start in range(0, len(items), batch_size):
        predictions.extend(
            engine.generate_batch(
                items[start : start + batch_size],
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
        )
        done = min(start + batch_size, len(items))
        if progress_every and (done % progress_every == 0 or done == len(items)):
            rate = done / max(time.time() - started, 1e-6)
            print(f"  [{name}] {done}/{len(items)} ({rate:.2f} ex/s)", flush=True)

    rows = []
    for ex, pred in zip(examples, predictions):
        row = evaluate_example(ex, pred)
        row["prediction"] = pred
        row["question"] = ex.question
        rows.append(row)

    return {
        "name": name,
        "count": len(rows),
        "seconds": round(time.time() - started, 1),
        "summary": aggregate(rows),
        "by_category": aggregate_by(rows, "category"),
        "by_answerability": aggregate_by(rows, "answerable"),
        "rows": rows,
    }


def strip_rows(result: dict) -> dict:
    """Result without the per-example rows, for compact JSON output."""
    return {k: v for k, v in result.items() if k != "rows"}


def write_predictions(path: str | Path, results: dict[str, dict]) -> None:
    import json

    from src.config import resolve_path

    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for suite, result in results.items():
            for row in result["rows"]:
                fh.write(json.dumps({"suite": suite, **row}, ensure_ascii=False) + "\n")
