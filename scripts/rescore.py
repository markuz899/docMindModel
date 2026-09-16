#!/usr/bin/env python
"""Re-run the evaluators over predictions saved by a previous run.

    python scripts/rescore.py artifacts/eval/baseline_results.json

Generation is the expensive part; scoring is not. When a metric definition
changes, rescore the stored answers instead of regenerating them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path
from src.evaluation.metrics import aggregate, aggregate_by, evaluate_example
from src.evaluation.runner import load_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="the results JSON written by src.evaluation.evaluate")
    parser.add_argument("--predictions", default=None,
                        help="defaults to <results stem>-predictions.jsonl")
    parser.add_argument("--out", default=None, help="defaults to overwriting --results")
    args = parser.parse_args()

    results_path = resolve_path(args.results)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    pred_path = resolve_path(
        args.predictions or results_path.with_name(results_path.stem + "-predictions.jsonl")
    )
    if not pred_path.exists():
        raise SystemExit(f"predictions not found: {pred_path}")

    by_suite: dict[str, dict[str, str]] = defaultdict(dict)
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_suite[row["suite"]][row["id"]] = row.get("prediction", "")

    for suite, predictions in by_suite.items():
        examples = {e.id: e for e in load_suite(suite)}
        missing = [i for i in predictions if i not in examples]
        if missing:
            print(f"[rescore] {suite}: {len(missing)} prediction(s) have no example, skipped")
        rows = [
            evaluate_example(examples[i], pred)
            for i, pred in predictions.items()
            if i in examples
        ]
        entry = payload.setdefault("suites", {}).setdefault(suite, {})
        entry["count"] = len(rows)
        entry["summary"] = aggregate(rows)
        entry["by_category"] = aggregate_by(rows, "category")
        entry["by_answerability"] = aggregate_by(rows, "answerable")
        print(f"[rescore] {suite}: {len(rows)} example(s)")

    payload.setdefault("environment", {})["rescored"] = True
    out = resolve_path(args.out or args.results)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
