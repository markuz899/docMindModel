#!/usr/bin/env python
"""Build and review a human-curated benchmark, held out of training forever.

    # sample candidates from a generated dataset, hitting the target mix
    python scripts/manual_benchmark.py build --from artifacts/corpus/real-dataset.jsonl --size 150

    # review them one by one: [a]pprove, [r]eject, [s]kip, [q]uit
    python scripts/manual_benchmark.py review

    # what is approved so far
    python scripts/manual_benchmark.py status

Approved items become benchmarks/manual/benchmark.jsonl. Every id that has ever
entered the review pool is recorded in benchmarks/manual/heldout_ids.txt, and
scripts/build_dataset.py refuses to put any of them into a training split --
including rejected ones, because a rejected candidate is still an example the
reviewer has read and judged against.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path
from src.dataset.io import read_jsonl, write_jsonl
from src.dataset.schema import Example

MANUAL_DIR = "benchmarks/manual"
CANDIDATES = f"{MANUAL_DIR}/candidates.jsonl"
APPROVED = f"{MANUAL_DIR}/benchmark.jsonl"
HELDOUT_IDS = f"{MANUAL_DIR}/heldout_ids.txt"
DECISIONS = f"{MANUAL_DIR}/decisions.json"

# What a useful benchmark looks like, per the brief.
TARGET_MIX = {"full": 0.50, "none": 0.25, "partial": 0.15, "noisy": 0.10}


def _bucket(example: Example) -> str:
    if example.answerable == "none":
        return "none"
    if example.answerable == "partial":
        return "partial"
    if "distractor" in example.tags or "contradiction" in example.tags:
        return "noisy"
    return "full"


def held_out_ids() -> set[str]:
    path = resolve_path(HELDOUT_IDS)
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def cmd_build(args) -> int:
    pool = read_jsonl(args.source)
    if not pool:
        raise SystemExit(f"no examples in {args.source}")
    rng = random.Random(args.seed)
    by_bucket: dict[str, list[Example]] = {}
    for example in pool:
        by_bucket.setdefault(_bucket(example), []).append(example)
    for items in by_bucket.values():
        rng.shuffle(items)

    selected: list[Example] = []
    shortfall: dict[str, int] = {}
    for bucket, share in TARGET_MIX.items():
        want = round(args.size * share)
        available = by_bucket.get(bucket, [])
        selected.extend(available[:want])
        if len(available) < want:
            shortfall[bucket] = want - len(available)

    rng.shuffle(selected)
    write_jsonl(CANDIDATES, selected)

    ids = held_out_ids() | {e.id for e in selected}
    path = resolve_path(HELDOUT_IDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")

    print(f"sampled {len(selected)} candidate(s) -> {CANDIDATES}")
    print("mix:", dict(Counter(_bucket(e) for e in selected)))
    if shortfall:
        print(f"WARNING: short of the target mix, the pool lacks: {shortfall}")
    print(f"{len(ids)} id(s) are now held out of training ({HELDOUT_IDS})")
    print(f"next: python scripts/manual_benchmark.py review")
    return 0


def _load_decisions() -> dict:
    path = resolve_path(DECISIONS)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_decisions(decisions: dict) -> None:
    path = resolve_path(DECISIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_review(args) -> int:
    candidates = read_jsonl(CANDIDATES)
    decisions = _load_decisions()
    pending = [e for e in candidates if e.id not in decisions]
    if not pending:
        print("nothing left to review")
        return cmd_status(args)

    print(f"{len(pending)} candidate(s) to review. "
          "[a]pprove  [r]eject  [s]kip  [q]uit\n")
    for index, example in enumerate(pending, 1):
        print("=" * 78)
        print(f"{index}/{len(pending)}  {example.id}  "
              f"[{example.answerable} / {example.category} / {example.project}]")
        print(f"\nQ: {example.question}\n")
        print(f"A: {example.answer}\n")
        print("sources in context:")
        for chunk in example.context:
            flag = " (distractor)" if chunk.relevant is False else ""
            print(f"  - [{chunk.source} — {chunk.heading}]{flag}")
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "q"
        if choice.startswith("q"):
            break
        if choice.startswith("a"):
            decisions[example.id] = {"approved": True}
        elif choice.startswith("r"):
            reason = input("reason: ").strip()
            decisions[example.id] = {"approved": False, "reason": reason}
        _save_decisions(decisions)

    approved = [e for e in candidates if decisions.get(e.id, {}).get("approved")]
    write_jsonl(APPROVED, approved)
    print(f"\nwrote {len(approved)} approved example(s) -> {APPROVED}")
    return 0


def cmd_status(args) -> int:
    candidates = read_jsonl(CANDIDATES) if resolve_path(CANDIDATES).exists() else []
    decisions = _load_decisions()
    approved = [e for e in candidates if decisions.get(e.id, {}).get("approved")]
    rejected = [e for e in candidates if decisions.get(e.id, {}).get("approved") is False]
    print(f"candidates : {len(candidates)}")
    print(f"approved   : {len(approved)}  {dict(Counter(_bucket(e) for e in approved))}")
    print(f"rejected   : {len(rejected)}")
    print(f"unreviewed : {len(candidates) - len(approved) - len(rejected)}")
    print(f"held out   : {len(held_out_ids())} id(s) excluded from training")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample candidates for review")
    build.add_argument("--from", dest="source", required=True)
    build.add_argument("--size", type=int, default=150)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=cmd_build)

    review = sub.add_parser("review", help="approve or reject candidates")
    review.set_defaults(func=cmd_review)

    status = sub.add_parser("status", help="show review progress")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
