#!/usr/bin/env python
"""Turn generated candidates into train/validation/test splits.

    python scripts/build_real_dataset.py --from artifacts/corpus/real-dataset.jsonl \
        --out-dir data/real --version 2.0.0

Held-out whole projects are the point: with enough projects the test set is
documentation the model has never seen, which is the only honest measure for
this task. Anything already reserved for the manual benchmark is excluded.

Writes to --out-dir rather than over data/train|validation|test, so the v1 demo
dataset -- and the published v1 results that depend on it -- stay intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config, resolve_path
from src.dataset.dedup import deduplicate
from src.dataset.io import read_jsonl, write_jsonl
from src.dataset.quality import check_example
from src.dataset.split import split_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_benchmark import held_out_ids  # noqa: E402


def summarise(examples) -> dict:
    return {
        "count": len(examples),
        "by_answerability": dict(Counter(str(e.answerable) for e in examples)),
        "by_category": dict(Counter(str(e.category) for e in examples).most_common()),
        "by_project": dict(Counter(str(e.project) for e in examples)),
        "with_distractors": sum(
            1 for e in examples if any(c.relevant is False for c in e.context)
        ),
        "avg_sources": round(sum(len(e.context) for e in examples) / max(len(examples), 1), 2),
        "avg_answer_chars": round(sum(len(e.answer) for e in examples) / max(len(examples), 1)),
        "teachers": dict(Counter(
            (e.teacher_metadata or {}).get("teacher_provider", "unknown") for e in examples
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="sources", nargs="+", required=True,
                        help="generated candidate JSONL file(s)")
    parser.add_argument("--out-dir", default="data/real")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    examples = [e for source in args.sources for e in read_jsonl(source)]
    if not examples:
        raise SystemExit(f"no examples in {args.sources}")
    print(f"loaded {len(examples)} candidate(s) from {len(args.sources)} file(s)")

    reserved = held_out_ids()
    if reserved:
        before = len(examples)
        examples = [e for e in examples if e.id not in reserved]
        print(f"held out {before - len(examples)} reserved for the manual benchmark")

    kept, rejected = [], []
    for example in examples:
        report = check_example(example, cfg.quality)
        (kept if report.ok else rejected).append(example if report.ok else report)
    print(f"quality gate: {len(kept)} passed, {len(rejected)} rejected")
    if rejected:
        reasons = Counter(i.split(":")[0].split("(")[0].strip()
                          for r in rejected for i in r.issues)
        for reason, count in reasons.most_common(6):
            print(f"  {count:5d}  {reason}")

    if not args.no_dedup:
        kept, dropped = deduplicate(kept, cfg.dedup)
        print(f"dedup: {len(kept)} kept, {len(dropped)} dropped")

    splits = split_dataset(kept, cfg.split, seed=42)
    total = sum(len(v) for v in splits.values())
    for name, items in splits.items():
        share = len(items) / max(total, 1)
        print(f"  {name:11s} {len(items):5d} ({share:.0%})  "
              f"projects={sorted({e.project for e in items})}")

    train_projects = {e.project for e in splits["train"]}
    test_projects = {e.project for e in splits["test"]}
    leak = train_projects & test_projects
    if leak:
        print(f"WARNING: projects in both train and test: {sorted(leak)}")
    elif not test_projects:
        print("WARNING: empty test split")
    else:
        print(f"test set is {len(test_projects)} entirely unseen project(s)")

    if args.dry_run:
        return 0

    out_dir = resolve_path(args.out_dir)
    paths = {name: out_dir / name / f"{name}.jsonl" for name in splits}
    for name, items in splits.items():
        write_jsonl(paths[name], items)

    card = {
        "dataset_version": args.version,
        "built_on": date.today().isoformat(),
        "sources": args.sources,
        "split_policy": cfg.split.model_dump(),
        "quality_gate": cfg.quality.model_dump(),
        "dedup": cfg.dedup.model_dump(),
        "rejected_by_quality_gate": len(rejected),
        "reserved_for_manual_benchmark": len(reserved),
        "totals": summarise(kept),
        "splits": {name: summarise(items) for name, items in splits.items()},
        "paths": {name: str(path) for name, path in paths.items()},
    }
    card_path = out_dir / "dataset_card.json"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote splits under {out_dir} and {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
