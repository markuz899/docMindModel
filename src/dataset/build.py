"""Build the demo dataset from the Markdown corpus + hand-written seeds.

Everything here is deterministic: the same corpus and seed files always produce
the same JSONL. Answers are never invented at build time -- seed answers are
hand-written and resolved against the real documentation, and the generated
examples are refusals, whose correct answer is a refusal by construction.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import yaml

from src.config import ChunkingConfig, DatasetConfig, resolve_path
from src.dataset.negatives import (
    EVAL_TEMPLATES,
    REFUSAL_TEMPLATES_EN,
    REFUSAL_TEMPLATES_IT,
    TRAIN_TEMPLATES,
    NegativeTemplate,
)
from src.dataset.quality import check_example
from src.dataset.schema import ContextChunk, Example
from src.generation.markdown import load_corpus
from src.text import normalize

SEED_DIR = "data/raw/seed"
DOCS_DIR = "data/raw/projects"


# --------------------------------------------------------------------------
# corpus index
# --------------------------------------------------------------------------

class CorpusIndex:
    def __init__(self, corpus: dict[str, list[ContextChunk]]) -> None:
        self.corpus = corpus
        self.by_key: dict[tuple[str, str, str], ContextChunk] = {}
        for project, chunks in corpus.items():
            for chunk in chunks:
                self.by_key[(project, chunk.source, chunk.heading)] = chunk

    def resolve(self, ref: str, default_project: str) -> ContextChunk:
        """``[project:]file.md#Heading`` -> chunk.

        The colon only introduces a project when a ``#`` follows it, so headings
        that contain a colon (``GET /users/:id``) resolve correctly.
        """
        head, sep, rest = ref.partition(":")
        project, ref_body = (head, rest) if sep and "#" in rest else (default_project, ref)
        source, _, heading = ref_body.partition("#")
        key = (project, source.strip(), heading.strip())
        if key not in self.by_key:
            raise KeyError(f"unknown documentation reference: {ref!r} (resolved to {key})")
        return self.by_key[key]

    def other_project_chunks(
        self, project: str, avoid: set[tuple[str, str]] | None = None
    ) -> list[ContextChunk]:
        """Chunks from other projects, minus any that collide on (source, heading).

        File names repeat across projects (`01-overview.md#Purpose` exists in all
        of them), and a citation token carries no project, so a colliding
        distractor would be indistinguishable from the real source.
        """
        avoid = avoid or set()
        return [
            c
            for p, chunks in self.corpus.items()
            if p != project
            for c in chunks
            if c.key not in avoid
        ]

    def projects(self) -> list[str]:
        return sorted(self.corpus)


def load_index(docs_dir: str | Path = DOCS_DIR) -> CorpusIndex:
    corpus = load_corpus(docs_dir, ChunkingConfig())
    if not corpus:
        raise SystemExit(f"no documentation found under {docs_dir}")
    return CorpusIndex(corpus)


# --------------------------------------------------------------------------
# seed examples
# --------------------------------------------------------------------------

def load_seed_specs(seed_dir: str | Path = SEED_DIR) -> list[dict]:
    specs = []
    for path in sorted(resolve_path(seed_dir).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        project = data.get("project")
        for item in data.get("examples", []):
            item = dict(item)
            item.setdefault("project", project)
            item["_file"] = path.name
            specs.append(item)
    return specs


def build_seed_examples(index: CorpusIndex, specs: Iterable[dict], seed: int = 42) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for spec in specs:
        project = spec["project"]
        relevant = [index.resolve(ref, project) for ref in spec["sources"]]
        distractors = [index.resolve(ref, project) for ref in spec.get("distractors", [])]
        chunks = [c.model_copy(update={"relevant": True}) for c in relevant]
        chunks += [c.model_copy(update={"relevant": False}) for c in distractors]
        rng.shuffle(chunks)
        examples.append(
            Example(
                id=spec["id"],
                question=spec["question"],
                context=chunks,
                answer=spec["answer"],
                category=spec.get("category"),
                difficulty=spec.get("difficulty"),
                answerable=spec.get("answerable", "full"),
                project=project,
                relevant_sources=[f"{c.source}#{c.heading}" for c in relevant],
                must_include=spec.get("must_include", []),
                must_not_include=spec.get("must_not_include", []),
                tags=list(spec.get("tags", [])) + ["seed"]
                + (["distractor"] if distractors else []),
                origin="seed",
            )
        )
    return examples


def build_distractor_variants(
    examples: list[Example], index: CorpusIndex, seed: int = 43, per_example: int = 1
) -> list[Example]:
    """Re-issue answerable examples with 3-4 irrelevant sources bolted on.

    The gold answer is unchanged, which is exactly the lesson: more noise in the
    context must not change the answer.
    """
    rng = random.Random(seed)
    variants: list[Example] = []
    for ex in examples:
        if ex.answerable == "none" or "distractor" in ex.tags:
            continue
        pool = index.other_project_chunks(ex.project or "", avoid=ex.context_keys())
        if len(pool) < 4:
            continue
        for n in range(per_example):
            noise = rng.sample(pool, k=rng.randint(3, 4))
            chunks = [c.model_copy() for c in ex.context]
            chunks += [c.model_copy(update={"relevant": False}) for c in noise]
            rng.shuffle(chunks)
            variants.append(
                ex.model_copy(
                    update={
                        "id": f"{ex.id}-noise{n + 1}",
                        "context": chunks,
                        "tags": sorted(set(ex.tags) | {"distractor", "augmented"}),
                        "source_count": len(chunks),
                    }
                )
            )
    return variants


# --------------------------------------------------------------------------
# generated unanswerable examples
# --------------------------------------------------------------------------

_GENERIC_HEADINGS = {"purpose", "overview", "scope", "runtime", "components", "layers"}


def _subject_for(chunks: list[ContextChunk], project: str) -> str:
    heading = chunks[0].heading
    return project if heading.lower() in _GENERIC_HEADINGS else heading


def _headings_phrase(chunks: list[ContextChunk], limit: int = 3) -> str:
    names = [f"«{c.heading}»" for c in chunks[:limit]]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " e " + names[-1]


def _headings_phrase_en(chunks: list[ContextChunk], limit: int = 3) -> str:
    names = [f'"{c.heading}"' for c in chunks[:limit]]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _context_is_silent(template: NegativeTemplate, chunks: list[ContextChunk]) -> bool:
    haystack = normalize(" ".join(f"{c.source} {c.heading} {c.content}" for c in chunks))
    return not any(normalize(p) in haystack for p in template.probes)


def build_negative_examples(
    index: CorpusIndex,
    templates: tuple[NegativeTemplate, ...],
    per_project: int,
    id_prefix: str,
    tags: list[str],
    seed: int = 44,
    projects: list[str] | None = None,
) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for project in projects or index.projects():
        chunks_all = index.corpus[project]
        made, attempts = 0, 0
        while made < per_project and attempts < per_project * 40:
            attempts += 1
            template = templates[(made + attempts) % len(templates)]
            size = rng.randint(2, 4)
            chunks = rng.sample(chunks_all, k=min(size, len(chunks_all)))
            if not _context_is_silent(template, chunks):
                continue
            italian = (made + attempts) % 3 != 2  # ~2/3 Italian, 1/3 English
            subject = _subject_for(chunks, project)
            if italian:
                question = template.question_it.format(subject=subject)
                body = REFUSAL_TEMPLATES_IT[made % len(REFUSAL_TEMPLATES_IT)].format(
                    topic=template.topic_it,
                    topic_cap=template.topic_it[:1].upper() + template.topic_it[1:],
                    subject=subject,
                    headings=_headings_phrase(chunks), need=template.need_it,
                )
            else:
                question = template.question_en.format(subject=subject)
                body = REFUSAL_TEMPLATES_EN[made % len(REFUSAL_TEMPLATES_EN)].format(
                    topic=template.topic_en,
                    topic_cap=template.topic_en[:1].upper() + template.topic_en[1:],
                    subject=subject,
                    headings=_headings_phrase_en(chunks), need=template.need_en,
                )
            ordered = [c.model_copy(update={"relevant": False}) for c in chunks]
            rng.shuffle(ordered)
            example = Example(
                id=f"{id_prefix}-{project}-{template.key}-{made:03d}",
                question=question,
                context=ordered,
                answer=body,
                category="bug_investigation" if template.key in {"root_cause"} else "how_it_works",
                difficulty="medium",
                answerable="none",
                project=project,
                relevant_sources=[],
                tags=sorted(set(tags) | {"unanswerable", f"probe:{template.key}"}),
                origin="synthetic",
            )
            out.append(example)
            made += 1
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_all(
    docs_dir: str | Path = DOCS_DIR,
    seed_dir: str | Path = SEED_DIR,
    cfg: DatasetConfig | None = None,
    negatives_per_project: int = 15,
    noise_variants: int = 1,
) -> dict[str, list[Example]]:
    cfg = cfg or DatasetConfig()
    index = load_index(docs_dir)

    seeds = build_seed_examples(index, load_seed_specs(seed_dir))
    noise = build_distractor_variants(seeds, index, per_example=noise_variants)
    negatives = build_negative_examples(
        index, TRAIN_TEMPLATES, negatives_per_project, "neg", ["generated"]
    )
    dataset = seeds + noise + negatives

    kept, rejected = [], []
    for ex in dataset:
        report = check_example(ex, cfg.quality)
        (kept if report.ok else rejected).append(ex if report.ok else report)

    return {"kept": kept, "rejected": rejected, "seeds": seeds, "noise": noise,
            "negatives": negatives}
