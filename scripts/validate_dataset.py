#!/usr/bin/env python
"""Validate JSONL dataset files against the schema and the quality gate.

    python scripts/validate_dataset.py --all
    python scripts/validate_dataset.py data/generated/candidates-2026-01-01.jsonl
    python scripts/validate_dataset.py --all --fix-out data/generated/clean.jsonl

Exits non-zero when anything fails, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config, resolve_path
from src.dataset.io import read_jsonl, write_jsonl
from src.dataset.quality import check_example

DEFAULT_FILES = [
    "data/train/train.jsonl",
    "data/validation/validation.jsonl",
    "data/test/test.jsonl",
    "benchmarks/hallucination.jsonl",
    "benchmarks/retrieval_noise.jsonl",
    "benchmarks/bug_investigation.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--all", action="store_true", help="validate the standard files")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--fix-out", default=None,
                        help="write only the passing examples to this file")
    parser.add_argument("--max-report", type=int, default=15)
    args = parser.parse_args()

    paths = list(args.paths) + (DEFAULT_FILES if args.all else [])
    if not paths:
        parser.error("give at least one path, or --all")

    cfg = load_dataset_config(args.config)
    failed_total = 0
    kept_all = []

    for path in paths:
        resolved = resolve_path(path)
        if not resolved.exists():
            print(f"{path}: MISSING")
            failed_total += 1
            continue
        examples = read_jsonl(path)
        issues, metrics = [], Counter()
        kept = []
        for ex in examples:
            report = check_example(ex, cfg.quality)
            if report.ok:
                kept.append(ex)
            else:
                issues.append((ex.id, report.issues))
                for issue in report.issues:
                    metrics[issue.split(":")[0].split("(")[0].strip()] += 1
        kept_all.extend(kept)
        status = "OK" if not issues else f"{len(issues)} FAILED"
        answerability = Counter(str(e.answerable) for e in examples)
        print(f"{path}: {len(examples)} example(s) -> {status} | {dict(answerability)}")
        for example_id, problems in issues[: args.max_report]:
            print(f"    - {example_id}: {'; '.join(problems)}")
        if len(issues) > args.max_report:
            print(f"    ... and {len(issues) - args.max_report} more")
        if metrics:
            print(f"    issue kinds: {dict(metrics)}")
        failed_total += len(issues)

    if args.fix_out:
        write_jsonl(args.fix_out, kept_all)
        print(f"wrote {len(kept_all)} passing example(s) to {args.fix_out}")

    print("\nRESULT:", "all files valid" if failed_total == 0 else f"{failed_total} failure(s)")
    return 0 if failed_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
