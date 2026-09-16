#!/usr/bin/env python
"""Build the demo dataset: corpus + seeds -> train/validation/test JSONL.

    python scripts/build_dataset.py [--negatives-per-project 15] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config, resolve_path
from src.dataset.build import build_all
from src.dataset.io import write_jsonl
from src.dataset.split import split_dataset


def summarise(examples) -> dict:
    def tally(attr):
        out: dict[str, int] = {}
        for ex in examples:
            key = str(getattr(ex, attr, None))
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    return {
        "count": len(examples),
        "by_category": tally("category"),
        "by_answerability": tally("answerable"),
        "by_project": tally("project"),
        "avg_sources": round(sum(len(e.context) for e in examples) / max(len(examples), 1), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--negatives-per-project", type=int, default=15)
    parser.add_argument("--noise-variants", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    built = build_all(
        cfg=cfg,
        negatives_per_project=args.negatives_per_project,
        noise_variants=args.noise_variants,
    )
    kept, rejected = built["kept"], built["rejected"]

    print(f"built {len(kept)} examples ({len(rejected)} rejected by the quality gate)")
    for report in rejected:
        print(f"  - {report.example_id}: {'; '.join(report.issues)}")

    splits = split_dataset(kept, cfg.split, seed=42)
    for name, items in splits.items():
        print(f"  {name:11s} {len(items):4d}  projects={sorted({e.project for e in items})}")

    if args.dry_run:
        return 0

    write_jsonl(f"{cfg.generated_dir}/dataset.jsonl", kept)
    write_jsonl(cfg.train_path, splits["train"])
    write_jsonl(cfg.validation_path, splits["validation"])
    write_jsonl(cfg.test_path, splits["test"])

    card = {
        "dataset_version": cfg.version,
        "built_on": date.today().isoformat(),
        "split_policy": cfg.split.model_dump(),
        "quality_gate": cfg.quality.model_dump(),
        "rejected": len(rejected),
        "totals": summarise(kept),
        "splits": {name: summarise(items) for name, items in splits.items()},
        "sources": {
            "seed_examples": len(built["seeds"]),
            "distractor_variants": len(built["noise"]),
            "generated_unanswerable": len(built["negatives"]),
        },
    }
    card_path = resolve_path(f"{cfg.generated_dir}/dataset_card.json")
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
