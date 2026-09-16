#!/usr/bin/env python
"""Generate candidate training examples from a folder of Markdown documentation.

    # no API key needed, exercises the whole pipeline
    python scripts/generate_dataset.py --docs path/to/docs --provider mock

    # real distillation
    export ANTHROPIC_API_KEY=...
    python scripts/generate_dataset.py --docs path/to/docs \
        --provider anthropic --model claude-opus-5 --max-contexts 200

Output is *candidate* data: review data/generated/candidates-*.jsonl (and the
matching -rejected.jsonl) before merging anything into the training set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config
from src.generation.providers import get_provider
from src.generation.teacher import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", required=True, help="documentation root (subdirs = projects)")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--provider", default=None, choices=["mock", "openai", "anthropic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--questions-per-context", type=int, default=None)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap contexts (cheap dry run)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    gen = cfg.generation
    overrides = {
        k: v
        for k, v in {
            "provider": args.provider,
            "model": args.model,
            "questions_per_context": args.questions_per_context,
            "max_contexts": args.max_contexts,
        }.items()
        if v is not None
    }
    cfg = cfg.model_copy(update={"generation": gen.model_copy(update=overrides)})

    provider = get_provider(
        cfg.generation.provider,
        model=cfg.generation.model,
        temperature=cfg.generation.temperature,
        max_output_tokens=cfg.generation.max_output_tokens,
    )
    generate_dataset(args.docs, cfg, provider, args.out, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
