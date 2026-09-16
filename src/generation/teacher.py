"""Knowledge-distillation pipeline: Markdown folders -> candidate JSONL.

markdown -> heading chunks -> context combinations -> teacher prompt ->
question/answer pairs -> quality filter -> reviewable JSONL.

Nothing here is bound to a specific teacher; see providers.py.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.config import DatasetConfig, GenerationPipelineConfig, QualityConfig, resolve_path
from src.dataset.quality import check_example
from src.dataset.schema import ContextChunk, Example, make_id
from src.generation.markdown import load_corpus
from src.generation.providers import LLMProvider, get_provider
from src.prompt import render_context

TEACHER_SYSTEM_PROMPT = """You build supervised training data for a small model \
that answers developer questions about a software project using ONLY the \
documentation sections it is given.

You will receive a set of documentation blocks, each introduced by its citation \
token in the form [file — heading].

Produce realistic questions a developer would actually ask, together with the \
ideal grounded answer.

Hard rules for every answer you write:
1. Use only facts present in the supplied blocks. Never add knowledge from \
   elsewhere, never invent class names, routes, tables, config keys or errors.
2. Cite sources inline with the exact [file — heading] token, copied verbatim.
3. Separate documented facts from hypotheses. A hypothesis must be labelled as one.
4. If the blocks cannot answer the question, say so explicitly instead of guessing.
5. Ignore blocks that are irrelevant to the question. Do not cite them.
6. If two blocks contradict each other, say the documentation is contradictory \
   and cite both. Do not pick a winner.
7. Write the answer in the same language as the question.
8. Be concise: a developer-useful answer, not an essay.

Return ONLY a JSON array. Each item:
{
  "question": str,
  "answer": str,
  "category": one of ["how_it_works","route","bug_investigation","dependencies",
                      "data_flow","database","configuration","integrations",
                      "errors","architecture","comparison","multi_source"],
  "difficulty": "easy" | "medium" | "hard",
  "answerability": "full" | "partial" | "none",
  "relevant_sources": ["<file>#<heading>", ...],
  "must_include": ["key fact 1", ...]
}
"""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class ContextBundle:
    project: str
    chunks: list[ContextChunk]
    relevant: list[ContextChunk]
    ask_unanswerable: bool


# --------------------------------------------------------------------------
# 3. context combinations
# --------------------------------------------------------------------------

def build_context_bundles(
    corpus: dict[str, list[ContextChunk]], cfg: GenerationPipelineConfig
) -> list[ContextBundle]:
    rng = random.Random(cfg.seed)
    projects = sorted(corpus)
    bundles: list[ContextBundle] = []

    for project in projects:
        chunks = corpus[project]
        others = [c for p in projects if p != project for c in corpus[p]]
        for anchor in chunks:
            size = rng.choice(cfg.context_sizes)
            pool = [c for c in chunks if c is not anchor]
            rng.shuffle(pool)
            relevant = [anchor] + pool[: max(0, size - 1)]
            selected = list(relevant)
            if others and rng.random() < cfg.distractor_ratio:
                distractors = rng.sample(others, k=min(len(others), rng.randint(1, 4)))
                selected += distractors
            relevant_ids = {id(c) for c in relevant}
            selected = [
                c.model_copy(update={"relevant": id(c) in relevant_ids}) for c in selected
            ]
            rng.shuffle(selected)
            bundles.append(
                ContextBundle(
                    project=project,
                    chunks=selected,
                    relevant=relevant,
                    ask_unanswerable=rng.random() < cfg.unanswerable_ratio,
                )
            )
    rng.shuffle(bundles)
    return bundles[: cfg.max_contexts]


# --------------------------------------------------------------------------
# 4. teacher prompt
# --------------------------------------------------------------------------

def build_teacher_prompt(bundle: ContextBundle, cfg: GenerationPipelineConfig) -> str:
    instruction = (
        f"Write {cfg.questions_per_context} question/answer pair(s) grounded in "
        "the blocks below."
    )
    if bundle.ask_unanswerable:
        instruction += (
            "\nAt least one pair MUST be a question that a developer would "
            "plausibly ask about this area but that the blocks CANNOT answer "
            '(answerability "none"). Its answer must state exactly what is '
            "missing, without guessing a cause."
        )
    if len(bundle.chunks) > len(bundle.relevant):
        instruction += (
            "\nSome blocks are irrelevant to the topic. Do not use or cite them."
        )
    return (
        f"{instruction}\n\n## Documentation blocks\n\n"
        f"{render_context([c.model_dump() for c in bundle.chunks])}\n"
    )


# --------------------------------------------------------------------------
# 5-6. generate + parse + validate
# --------------------------------------------------------------------------

def parse_teacher_output(raw: str) -> list[dict]:
    """Tolerate fenced blocks and leading prose around the JSON array."""
    if not raw or not raw.strip():
        return []
    candidates = [m.group(1) for m in _JSON_BLOCK_RE.finditer(raw)] or []
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    candidates.append(raw)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed = parsed.get("examples") or parsed.get("items") or [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def item_to_example(item: dict, bundle: ContextBundle, origin: str) -> Example | None:
    question, answer = (item.get("question") or "").strip(), (item.get("answer") or "").strip()
    if not question or not answer:
        return None
    answerability = (item.get("answerability") or "full").lower()
    if answerability not in {"full", "partial", "none"}:
        answerability = "full"
    category = (item.get("category") or "how_it_works").lower()
    difficulty = (item.get("difficulty") or "medium").lower()
    try:
        return Example(
            id=make_id(f"gen-{bundle.project}", question, answer),
            question=question,
            context=[c.model_copy() for c in bundle.chunks],
            answer=answer,
            category=category,
            difficulty=difficulty,
            answerable=answerability,
            project=bundle.project,
            relevant_sources=[str(s) for s in item.get("relevant_sources", []) if s],
            must_include=[str(s) for s in item.get("must_include", []) if s],
            tags=(["distractor"] if len(bundle.chunks) > len(bundle.relevant) else []),
            origin=origin,
        )
    except Exception as exc:  # pydantic validation of enums / lengths
        print(f"[teacher] dropped malformed item: {exc}")
        return None


def generate_dataset(
    docs_dir: str | Path,
    dataset_cfg: DatasetConfig | None = None,
    provider: LLMProvider | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Run the whole pipeline and write reviewable candidate/rejected JSONL."""
    from src.dataset.io import write_jsonl

    dataset_cfg = dataset_cfg or DatasetConfig()
    gen_cfg = dataset_cfg.generation
    quality_cfg = dataset_cfg.quality

    corpus = load_corpus(docs_dir, dataset_cfg.chunking)
    if not corpus:
        raise SystemExit(f"no markdown found under {docs_dir}")

    provider = provider or get_provider(
        gen_cfg.provider,
        model=gen_cfg.model,
        temperature=gen_cfg.temperature,
        max_output_tokens=gen_cfg.max_output_tokens,
    )
    origin = f"teacher:{provider.name}:{getattr(provider, 'model', '?')}"

    bundles = build_context_bundles(corpus, gen_cfg)
    if limit:
        bundles = bundles[:limit]

    accepted: list[Example] = []
    rejected: list[dict] = []
    for i, bundle in enumerate(bundles, 1):
        prompt = build_teacher_prompt(bundle, gen_cfg)
        try:
            raw = provider.complete(TEACHER_SYSTEM_PROMPT, prompt)
        except Exception as exc:  # network / rate limits should not lose prior work
            print(f"[teacher] bundle {i}/{len(bundles)} failed: {exc}")
            continue
        for item in parse_teacher_output(raw):
            example = item_to_example(item, bundle, origin)
            if example is None:
                continue
            report = check_example(example, quality_cfg)
            if report.ok:
                accepted.append(example)
            else:
                rejected.append(
                    {"id": example.id, "question": example.question,
                     "answer": example.answer, "issues": report.issues,
                     "metrics": report.metrics}
                )
        if i % 10 == 0:
            print(f"[teacher] {i}/{len(bundles)} contexts, {len(accepted)} accepted")

    stamp = date.today().isoformat()
    out = resolve_path(output_path or f"{dataset_cfg.generated_dir}/candidates-{stamp}.jsonl")
    rej = out.with_name(out.stem + "-rejected.jsonl")
    write_jsonl(out, accepted)
    rej.parent.mkdir(parents=True, exist_ok=True)
    rej.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    summary = {
        "contexts": len(bundles),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "provider": origin,
        "candidates_path": str(out),
        "rejected_path": str(rej),
    }
    print(json.dumps(summary, indent=2))
    return summary
