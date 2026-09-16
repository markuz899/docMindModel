#!/usr/bin/env python
"""Compare the base model and the fine-tuned model on identical test data.

    python scripts/evaluate_models.py --adapter artifacts/adapter

Both models see the same suites, the same prompts and the same decoding
settings; the only difference is the adapter. Models are loaded one at a time
so this runs on a machine that can hold exactly one of them.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_model_config, resolve_path
from src.evaluation.runner import SUITES, load_suite, run_suite, strip_rows
from src.inference.engine import DocMindEngine

METRICS = [
    ("hallucinated_rate", "hallucination", "lower"),
    ("refusal_correct_rate", "refusal", "higher"),
    ("answer_correctness", "correctness", "higher"),
    ("groundedness", "grounded", "higher"),
    ("citation_f1", "citation F1", "higher"),
    ("relevant_source_usage", "rel. sources", "higher"),
    ("conciseness", "concise", "higher"),
]


def evaluate(model_config: str, adapter: str | None, label: str, suites: list[str],
             limit: int | None, batch_size: int, max_new_tokens: int | None) -> dict:
    engine = DocMindEngine.load(model_config, adapter, label)
    print(f"\n### {engine.label} on {engine.device}")
    results = {}
    for name in suites:
        examples = load_suite(name, limit)
        results[name] = strip_rows(
            run_suite(engine, examples, batch_size=batch_size,
                      max_new_tokens=max_new_tokens, name=name)
        )
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def load_results(path: str, suites: list[str]) -> dict:
    """Reuse an earlier --out file instead of regenerating the same answers."""
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    stored = payload.get("suites", payload)
    missing = [s for s in suites if s not in stored]
    if missing:
        raise SystemExit(f"{path} has no results for {missing}; rerun without --*-results")
    return {name: stored[name] for name in suites}


def print_comparison(base: dict, tuned: dict) -> None:
    for suite in base:
        print(f"\n{suite}  (n={base[suite]['count']})")
        print(f"  {'metric':<16}{'base':>10}{'fine-tuned':>13}{'delta':>10}   better")
        print("  " + "-" * 60)
        for key, label, direction in METRICS:
            b = base[suite]["summary"].get(key)
            t = tuned[suite]["summary"].get(key)
            if b is None or t is None:
                continue
            delta = t - b
            improved = delta < 0 if direction == "lower" else delta > 0
            mark = "  ✓" if improved and abs(delta) > 1e-6 else ("  ✗" if abs(delta) > 1e-6 else "  =")
            print(f"  {label:<16}{b:>10.4f}{t:>13.4f}{delta:>+10.4f}{mark}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default="artifacts/adapter")
    parser.add_argument("--suites", default="test,hallucination,retrieval_noise,bug_investigation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out", default="artifacts/eval/comparison.json")
    parser.add_argument("--base-results", default=None,
                        help="reuse a previous baseline results JSON instead of rerunning")
    parser.add_argument("--tuned-results", default=None,
                        help="reuse a previous fine-tuned results JSON instead of rerunning")
    args = parser.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITES and not Path(s).exists()]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}")
    if not args.tuned_results and not resolve_path(args.adapter).exists():
        raise SystemExit(f"adapter not found: {args.adapter} -- train first")

    common = dict(suites=suites, limit=args.limit, batch_size=args.batch_size,
                  max_new_tokens=args.max_new_tokens)
    base = (
        load_results(args.base_results, suites)
        if args.base_results
        else evaluate(args.model_config, None, "base", **common)
    )
    tuned = (
        load_results(args.tuned_results, suites)
        if args.tuned_results
        else evaluate(args.model_config, args.adapter, "fine-tuned", **common)
    )

    print_comparison(base, tuned)

    payload = {
        "compared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": load_model_config(args.model_config).base_model,
        "adapter": args.adapter,
        "limit": args.limit,
        "base": base,
        "fine_tuned": tuned,
        "deltas": {
            suite: {
                key: round(tuned[suite]["summary"].get(key, 0) - base[suite]["summary"].get(key, 0), 4)
                for key, _, _ in METRICS
                if key in base[suite]["summary"]
            }
            for suite in base
        },
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
