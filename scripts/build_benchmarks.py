#!/usr/bin/env python
"""Build the three targeted benchmarks under benchmarks/.

  hallucination.jsonl    >=100 questions the context cannot answer
  retrieval_noise.jsonl  1 relevant source + 4 irrelevant ones
  bug_investigation.jsonl hand-written debugging scenarios with a rubric

The hallucination benchmark uses EVAL_TEMPLATES, which share no topic with the
TRAIN_TEMPLATES used by the training set: measuring refusal with the same
phrasings the model was trained on would measure memorisation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config, resolve_path
from src.dataset.build import CorpusIndex, build_negative_examples, load_index, load_seed_specs, build_seed_examples
from src.dataset.io import write_jsonl
from src.dataset.negatives import EVAL_TEMPLATES
from src.dataset.quality import check_example
from src.dataset.schema import Example

BUG_YAML = "data/raw/benchmarks/bug_investigation.yaml"


def build_hallucination(index: CorpusIndex, target: int) -> list[Example]:
    # margin: a few templates get skipped when the sampled context mentions them
    per_project = -(-target // len(index.projects())) + 5
    items = build_negative_examples(
        index, EVAL_TEMPLATES, per_project, "halluc", ["benchmark"], seed=99
    )
    return items


def build_retrieval_noise(
    index: CorpusIndex, held_out: set[str], distractors: int = 4, seed: int = 101
) -> list[Example]:
    """Answerable seeds re-issued with exactly N irrelevant sources added."""
    rng = random.Random(seed)
    seeds = build_seed_examples(index, load_seed_specs())
    out: list[Example] = []
    for ex in seeds:
        if ex.answerable == "none" or "distractor" in ex.tags:
            continue
        pool = index.other_project_chunks(ex.project or "", avoid=ex.context_keys())
        noise = rng.sample(pool, k=distractors)
        chunks = [c.model_copy() for c in ex.context]
        chunks += [c.model_copy(update={"relevant": False}) for c in noise]
        rng.shuffle(chunks)
        out.append(
            ex.model_copy(
                update={
                    "id": f"noise-{ex.id}",
                    "context": chunks,
                    "source_count": len(chunks),
                    "tags": sorted(
                        set(ex.tags) | {"benchmark", "retrieval_noise"}
                        | ({"heldout"} if ex.project in held_out else {"in_domain"})
                    ),
                }
            )
        )
    return out


def build_bug_benchmark(index: CorpusIndex) -> list[Example]:
    data = yaml.safe_load(resolve_path(BUG_YAML).read_text(encoding="utf-8"))
    out: list[Example] = []
    for spec in data["examples"]:
        project = spec["project"]
        chunks = [index.resolve(ref, project) for ref in spec["sources"]]
        out.append(
            Example(
                id=spec["id"],
                question=spec["question"],
                context=[c.model_copy(update={"relevant": True}) for c in chunks],
                answer=spec["answer"],
                category=spec.get("category", "bug_investigation"),
                difficulty=spec.get("difficulty"),
                answerable=spec.get("answerable", "partial"),
                project=project,
                relevant_sources=[f"{c.source}#{c.heading}" for c in chunks],
                must_include=spec.get("must_include", []),
                must_not_include=spec.get("must_not_include", []),
                tags=["benchmark", "bug_investigation"]
                + (["heldout"] if spec.get("heldout") else ["in_domain"]),
                origin="seed",
            )
        )
    return out


def _validate(name: str, items: list[Example], cfg) -> list[Example]:
    ok, bad = [], []
    for ex in items:
        report = check_example(ex, cfg.quality)
        (ok if report.ok else bad).append(ex if report.ok else report)
    for report in bad:
        print(f"  [{name}] rejected {report.example_id}: {'; '.join(report.issues)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hallucination-target", type=int, default=100)
    parser.add_argument("--held-out-project", default="mesh-gateway",
                        help="project reserved for the test split")
    args = parser.parse_args()

    cfg = load_dataset_config()
    index = load_index()
    out_dir = resolve_path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    halluc = _validate("hallucination", build_hallucination(index, args.hallucination_target), cfg)
    noise = _validate("retrieval_noise",
                      build_retrieval_noise(index, {args.held_out_project}), cfg)
    bugs = _validate("bug_investigation", build_bug_benchmark(index), cfg)

    if len(halluc) < args.hallucination_target:
        print(f"WARNING: only {len(halluc)} hallucination items "
              f"(target {args.hallucination_target})")

    for name, items in (("hallucination", halluc), ("retrieval_noise", noise),
                        ("bug_investigation", bugs)):
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, items)
        print(f"{name:20s} {len(items):4d} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
