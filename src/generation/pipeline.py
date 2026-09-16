"""Real-documentation generation: corpus -> planned batches -> teacher -> JSONL.

This is the single orchestration path (the demo `mock` teacher runs through it
too). What it adds over a naive loop:

  * **batching** -- one CLI invocation yields several examples;
  * **caching** -- an identical request is never sent twice, across runs;
  * **resume** -- a run killed by a quota limit continues where it stopped;
  * **planning** -- the answerability/category mix is decided up front rather
    than hoped for;
  * **audit** -- every example records which teacher, which CLI version and
    which request produced it.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import DatasetConfig, GenerationPipelineConfig, resolve_path
from src.dataset.io import write_jsonl
from src.dataset.quality import check_example
from src.dataset.schema import ContextChunk, Example, make_id
from src.generation.cache import PROMPT_VERSION, RunState, TeacherCache
from src.generation.cli_teachers import TeacherCallError, TeacherUsageLimit
from src.generation.markdown import load_corpus
from src.generation.prompts import (
    SYSTEM_PROMPT,
    TEACHER_JSON_SCHEMA,
    build_batch_instruction,
)
from src.generation.teacher import parse_teacher_output
from src.prompt import render_context

# Categories usable for answerable questions, with bug_investigation weighted up:
# it is DocMind's primary use case, so the dataset should over-represent it.
ANSWERABLE_CATEGORIES = (
    "how_it_works", "api", "api", "service", "dependency", "architecture",
    "database", "data_flow", "configuration", "integration", "error_handling",
    "bug_investigation", "bug_investigation", "bug_investigation", "multi_source",
)


@dataclass
class BundlePlan:
    """One teacher call: a context, and the mix of examples to ask for."""

    id: str
    project: str
    chunks: list[ContextChunk]
    relevant: list[ContextChunk]
    slots: list[tuple[str, str]]  # (answerability, category)
    has_contradiction: bool = False

    @property
    def has_distractors(self) -> bool:
        return len(self.chunks) > len(self.relevant)


@dataclass
class RunReport:
    teacher: str
    teacher_version: str | None
    prompt_version: str
    started_at: str
    bundles_planned: int = 0
    bundles_done: int = 0
    teacher_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    examples_accepted: int = 0
    examples_rejected: int = 0
    parse_failures: int = 0
    call_failures: int = 0
    usage_limit_hit: bool = False
    seconds: float = 0.0
    stopped_early: str | None = None

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["seconds"] = round(self.seconds, 1)
        return data


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def _draw_slot(rng: random.Random, weights: dict[str, float]) -> tuple[str, str]:
    kinds, probs = zip(*weights.items())
    kind = rng.choices(kinds, weights=probs, k=1)[0]
    if kind == "none":
        return "none", "unanswerable"
    if kind == "partial":
        return "partial", "partially_answerable"
    if kind == "noisy":
        return "full", "contradictory_context"
    return "full", rng.choice(ANSWERABLE_CATEGORIES)


def plan_bundles(
    corpus: dict[str, list[ContextChunk]], cfg: GenerationPipelineConfig
) -> list[BundlePlan]:
    """Turn a corpus into a deterministic list of teacher calls."""
    rng = random.Random(cfg.seed)
    projects = sorted(corpus)
    weights = cfg.answerability.weights()
    bundles: list[BundlePlan] = []

    for project in projects:
        chunks = corpus[project]
        own_keys = {c.key for c in chunks}
        others = [
            c for p in projects if p != project
            for c in corpus[p] if c.key not in own_keys
        ]
        for anchor in chunks:
            size = rng.choice(cfg.context_sizes)
            pool = [c for c in chunks if c is not anchor]
            rng.shuffle(pool)
            relevant = [anchor] + pool[: max(0, size - 1)]
            selected = list(relevant)
            if others and rng.random() < cfg.distractor_ratio:
                selected += rng.sample(others, k=min(len(others), rng.randint(2, 4)))
            relevant_ids = {id(c) for c in relevant}
            selected = [c.model_copy(update={"relevant": id(c) in relevant_ids})
                        for c in selected]
            rng.shuffle(selected)

            slots = [_draw_slot(rng, weights) for _ in range(cfg.examples_per_teacher_call)]
            has_contradiction = (
                rng.random() < cfg.contradiction_ratio and len(relevant) >= 2
            )
            if has_contradiction:
                slots[0] = ("partial", "contradictory_context")

            bundles.append(
                BundlePlan(
                    id=make_id(f"b-{project}", anchor.source, anchor.heading,
                               str(len(selected)), str(slots)),
                    project=project,
                    chunks=selected,
                    relevant=relevant,
                    slots=slots,
                    has_contradiction=has_contradiction,
                )
            )
    rng.shuffle(bundles)
    return bundles[: cfg.max_contexts]


def build_prompt(bundle: BundlePlan) -> str:
    instruction = build_batch_instruction(
        len(bundle.slots), bundle.slots, bundle.has_distractors, bundle.has_contradiction
    )
    blocks = render_context([c.model_dump() for c in bundle.chunks])
    return f"{instruction}\n\n## Documentation blocks\n\n{blocks}\n"


# --------------------------------------------------------------------------
# mapping teacher output -> Example
# --------------------------------------------------------------------------

def item_to_example(item: dict, bundle: BundlePlan, audit: dict) -> Example | None:
    question = (item.get("question") or "").strip()
    answer = (item.get("answer") or "").strip()
    if not question or not answer:
        return None
    answerable = (item.get("answerable") or item.get("answerability") or "full").lower()
    if answerable not in {"full", "partial", "none"}:
        answerable = "full"
    # The teacher schema calls it required_sources; the dataset calls the same
    # thing relevant_sources. Map, do not store twice.
    required = [
        str(s).strip()
        for s in (item.get("required_sources") or item.get("relevant_sources") or [])
        if s
    ]
    try:
        return Example(
            id=make_id(f"gen-{bundle.project}", question, answer),
            question=question,
            context=[c.model_copy() for c in bundle.chunks],
            answer=answer,
            category=(item.get("category") or "how_it_works").lower(),
            difficulty=(item.get("difficulty") or "medium").lower(),
            answerable=answerable,
            project=bundle.project,
            relevant_sources=required,
            facts=[str(f) for f in (item.get("facts") or []) if f],
            unsupported_claims=[str(f) for f in (item.get("unsupported_claims") or []) if f],
            tags=(["distractor"] if bundle.has_distractors else [])
            + (["contradiction"] if bundle.has_contradiction else []),
            origin=f"teacher:{audit['teacher_provider']}",
            teacher_metadata=audit,
        )
    except Exception as exc:
        print(f"[gen] dropped malformed item: {exc}")
        return None


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def run_generation(
    docs_dir: str | Path,
    teacher,
    dataset_cfg: DatasetConfig | None = None,
    output_path: str | Path | None = None,
    resume: bool = False,
    max_bundles: int | None = None,
    project_name: str | None = None,
) -> tuple[list[Example], RunReport]:
    dataset_cfg = dataset_cfg or DatasetConfig()
    cfg = dataset_cfg.generation

    corpus = load_corpus(docs_dir, dataset_cfg.chunking, project_name=project_name)
    if not corpus:
        raise SystemExit(f"no documentation found under {docs_dir}")

    out_path = resolve_path(output_path or f"{dataset_cfg.generated_dir}/real-candidates.jsonl")
    state_path = out_path.with_name(out_path.stem + ".state.json")
    rejects_path = out_path.with_name(out_path.stem + "-rejected.jsonl")

    cache = TeacherCache(root=resolve_path(cfg.cache_dir), enabled=cfg.cache_enabled)
    state = RunState.load(state_path, resume=resume)

    bundles = plan_bundles(corpus, cfg)
    if max_bundles:
        bundles = bundles[:max_bundles]
    if cfg.max_teacher_calls:
        bundles = bundles[: cfg.max_teacher_calls]

    version = teacher.provider_version()
    report = RunReport(
        teacher=teacher.provider_name(),
        teacher_version=version,
        prompt_version=PROMPT_VERSION,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        bundles_planned=len(bundles),
    )

    accepted: list[Example] = []
    rejected: list[dict] = []
    if resume and out_path.exists():
        accepted = [Example(**json.loads(line))
                    for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"[gen] resuming: {len(accepted)} example(s) already on disk, "
              f"{len(state.completed)} bundle(s) already done")

    pending = [b for b in bundles if not state.done(b.id)]
    print(f"[gen] teacher={report.teacher} ({version}) | {len(pending)} bundle(s) to run "
          f"of {len(bundles)} planned | batch={cfg.examples_per_teacher_call} "
          f"| concurrency={cfg.concurrency}")

    schema = TEACHER_JSON_SCHEMA if teacher.supports_structured_output() else None
    started = time.time()

    def process(bundle: BundlePlan) -> tuple[BundlePlan, str | None, bool, str | None]:
        """-> (bundle, raw_response, from_cache, error). Never raises."""
        user = build_prompt(bundle)
        key = TeacherCache.request_hash(
            teacher.provider_name(), version, getattr(teacher, "model", None),
            SYSTEM_PROMPT, user, schema,
            options={"slots": bundle.slots},
        )
        cached = cache.get(key)
        if cached is not None:
            return bundle, cached, True, None
        try:
            raw = teacher.complete(SYSTEM_PROMPT, user, schema)
        except TeacherUsageLimit as exc:
            return bundle, None, False, f"USAGE_LIMIT:{exc}"
        except TeacherCallError as exc:
            return bundle, None, False, str(exc)
        cache.put(key, raw, {
            "teacher": teacher.provider_name(), "version": version,
            "project": bundle.project, "bundle": bundle.id,
        })
        return bundle, raw, False, None

    def absorb(bundle: BundlePlan, raw: str, from_cache: bool, request_hash: str) -> None:
        items = parse_teacher_output(raw)
        if not items:
            report.parse_failures += 1
            rejected.append({"bundle": bundle.id, "issues": ["teacher output did not parse"],
                             "raw": (raw or "")[:800]})
            return
        audit = {
            "teacher_provider": teacher.provider_name(),
            "teacher_cli_version": version,
            "teacher_model": getattr(teacher, "model", None),
            "prompt_version": PROMPT_VERSION,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_hash": bundle.id,
            "teacher_request_hash": request_hash,
            "from_cache": from_cache,
        }
        for item in items:
            example = item_to_example(item, bundle, audit)
            if example is None:
                report.examples_rejected += 1
                continue
            check = check_example(example, dataset_cfg.quality)
            if check.ok:
                accepted.append(example)
                report.examples_accepted += 1
            else:
                report.examples_rejected += 1
                rejected.append({"id": example.id, "bundle": bundle.id,
                                 "question": example.question, "answer": example.answer,
                                 "issues": check.issues, "metrics": check.metrics})

    def checkpoint() -> None:
        write_jsonl(out_path, accepted)
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
        rejects_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected)
            + ("\n" if rejected else ""), encoding="utf-8")
        state.counters.update(report.as_dict())
        state.save()

    try:
        worker = max(1, min(cfg.concurrency, 2))  # quota and reproducibility first
        with ThreadPoolExecutor(max_workers=worker) as pool:
            for i, (bundle, raw, from_cache, error) in enumerate(
                pool.map(process, pending) if worker > 1 else map(process, pending), 1
            ):
                if error and error.startswith("USAGE_LIMIT:"):
                    report.usage_limit_hit = True
                    report.stopped_early = error[len("USAGE_LIMIT:"):]
                    break
                if error:
                    report.call_failures += 1
                    print(f"[gen] bundle {bundle.id} failed: {error[:200]}")
                    continue
                if from_cache:
                    report.cache_hits += 1
                else:
                    report.cache_misses += 1
                    report.teacher_calls += 1
                user = build_prompt(bundle)
                request_hash = TeacherCache.request_hash(
                    teacher.provider_name(), version, getattr(teacher, "model", None),
                    SYSTEM_PROMPT, user, schema, options={"slots": bundle.slots})
                absorb(bundle, raw or "", from_cache, request_hash)
                state.mark(bundle.id)
                report.bundles_done += 1
                if i % 10 == 0:
                    checkpoint()
                    rate = report.examples_accepted / max(time.time() - started, 1e-6)
                    print(f"[gen] {i}/{len(pending)} bundles | "
                          f"{report.examples_accepted} accepted | "
                          f"{report.cache_hits} cache hits | {rate:.2f} ex/s")
    except KeyboardInterrupt:
        report.stopped_early = "interrupted by user (SIGINT)"
        print("\n[gen] interrupted -- checkpointing before exit")

    report.seconds = time.time() - started
    report.cache_hits = cache.stats.hits
    report.cache_misses = cache.stats.misses
    checkpoint()

    print("\n" + json.dumps(report.as_dict(), indent=2))
    if report.usage_limit_hit:
        print(
            f"\nTeacher usage limit reached.\n"
            f"  Generated: {report.examples_accepted} example(s)\n"
            f"  Bundles:   {report.bundles_done}/{report.bundles_planned}\n"
            f"  Cached:    {report.cache_hits} hit(s)\n"
            f"  Progress saved to {out_path}\n\n"
            f"Re-run the same command with --resume once your quota resets."
        )
    return accepted, report
