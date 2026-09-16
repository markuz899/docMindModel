#!/usr/bin/env python
"""Evaluate one model on the test set and the targeted benchmarks.

    # baseline, before any training
    python -m src.evaluation.evaluate --out artifacts/eval/baseline_results.json

    # after training
    python -m src.evaluation.evaluate --adapter artifacts/adapter \
        --out artifacts/eval/finetuned_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_model_config, resolve_path
from src.evaluation.runner import SUITES, load_suite, run_suite, strip_rows, write_predictions
from src.inference.engine import DocMindEngine

HEADLINE = (
    "answer_correctness", "groundedness", "citation_f1", "refusal_correct_rate",
    "hallucinated_rate", "relevant_source_usage", "conciseness",
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def print_table(results: dict[str, dict]) -> None:
    width = max(len(name) for name in results) + 2
    header = f"{'suite':<{width}}" + "".join(f"{k[:14]:>16}" for k in HEADLINE)
    print("\n" + header)
    print("-" * len(header))
    for name, result in results.items():
        summary = result["summary"]
        row = f"{name:<{width}}" + "".join(
            f"{summary.get(k, float('nan')):>16.4f}" for k in HEADLINE
        )
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--suites", default="test,hallucination,retrieval_noise,bug_investigation")
    parser.add_argument("--limit", type=int, default=None, help="examples per suite")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", default="artifacts/eval/results.json")
    parser.add_argument("--predictions", default=None,
                        help="also dump every prediction as JSONL (default: next to --out)")
    args = parser.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITES and not Path(s).exists()]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}; known: {sorted(SUITES)}")

    model_cfg = load_model_config(args.model_config)
    engine = DocMindEngine.load(model_cfg, args.adapter, args.label)
    print(f"model: {engine.label} | device: {engine.device}")

    results: dict[str, dict] = {}
    for name in suites:
        examples = load_suite(name, args.limit)
        print(f"\n== {name}: {len(examples)} example(s)")
        results[name] = run_suite(
            engine, examples, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, name=name,
        )

    print_table(results)

    payload = {
        "model": {
            "label": engine.label,
            "base_model": model_cfg.base_model,
            "adapter": args.adapter,
            "max_seq_length": model_cfg.max_seq_length,
            "generation": model_cfg.generation.model_dump(),
        },
        "environment": {
            "device": engine.device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_sha": _git_sha(),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "limit": args.limit,
        "suites": {name: strip_rows(result) for name, result in results.items()},
    }
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")

    pred_path = args.predictions or out_path.with_name(out_path.stem + "-predictions.jsonl")
    write_predictions(pred_path, results)
    print(f"wrote {resolve_path(pred_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
