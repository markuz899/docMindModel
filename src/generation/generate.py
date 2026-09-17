#!/usr/bin/env python
"""Generate training data from real documentation using a local teacher CLI.

    # what is installed and authenticated, and would anything be billed?
    python -m src.generation.generate --health-check

    # generate, using whichever local CLI is available
    python -m src.generation.generate --docs /path/to/docs --teacher auto

    # continue after a usage limit
    python -m src.generation.generate --docs /path/to/docs --teacher codex --resume

The teacher runs read-only with no tools and sees only the documentation blocks
of the current batch -- never this repository. There is no fallback to a metered
API: if the subscription CLI cannot run, the command fails and says why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_dataset_config, resolve_path
from src.dataset.dedup import deduplicate
from src.dataset.io import write_jsonl
from src.generation.markdown import DOC_SUFFIXES, load_corpus
from src.generation.pipeline import run_generation
from src.generation.providers import health_report_all, select_teacher


def summarise_corpus(docs: str, cfg) -> dict:
    corpus = load_corpus(docs, cfg.chunking)
    return {
        "projects": len(corpus),
        "sections": sum(len(v) for v in corpus.values()),
        "per_project": {k: len(v) for k, v in sorted(corpus.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--docs", help="documentation root; subdirectories become projects")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--teacher", default=None,
                        choices=["auto", "codex", "claude", "mock", "selfhosted",
                                 "openai", "anthropic"])
    parser.add_argument("--model", default=None, help="override the CLI's default model")
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue a previous run instead of starting over")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="examples requested per teacher call")
    parser.add_argument("--max-bundles", type=int, default=None,
                        help="cap teacher calls this run (cheap trial run)")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="seconds per teacher call")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--project-name", default=None,
                        help="project label for docs sitting directly in --docs")
    parser.add_argument("--allow-metered-env", action="store_true",
                        help="accept per-token billing: keeps metered credentials in the "
                             "teacher's environment and skips the billing block")
    parser.add_argument("--health-check", action="store_true",
                        help="report teacher CLI availability and billing risk, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and report, contact no teacher")
    args = parser.parse_args()

    if args.health_check:
        reports = health_report_all()
        for report in reports:
            print(report.render())
            print()
        usable = [r for r in reports if r.usable and not r.metered_blocking]
        print(f"Usable on a subscription: {', '.join(r.provider for r in usable) or 'none'}")
        return 0 if usable else 1

    if not args.docs:
        parser.error("--docs is required (or use --health-check)")

    cfg = load_dataset_config(args.config)
    overrides = {
        k: v for k, v in {
            "teacher": args.teacher,
            "model": args.model,
            "examples_per_teacher_call": args.batch_size,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout,
            "cache_enabled": False if args.no_cache else None,
        }.items() if v is not None
    }
    cfg = cfg.model_copy(
        update={"generation": cfg.generation.model_copy(update=overrides)}
    )
    gen = cfg.generation

    docs_root = resolve_path(args.docs)
    if not docs_root.exists():
        raise SystemExit(f"--docs path does not exist: {docs_root}")

    corpus_summary = summarise_corpus(str(docs_root), cfg)
    if not corpus_summary["sections"]:
        raise SystemExit(
            f"no documentation found under {docs_root} "
            f"(looking for {', '.join(DOC_SUFFIXES)})"
        )
    print(f"corpus: {corpus_summary['projects']} project(s), "
          f"{corpus_summary['sections']} section(s)")
    for project, count in corpus_summary["per_project"].items():
        print(f"  {project:<40} {count:5d} sections")

    if args.dry_run:
        from src.generation.pipeline import plan_bundles

        bundles = plan_bundles(load_corpus(str(docs_root), cfg.chunking), gen)
        planned = len(bundles) * gen.examples_per_teacher_call
        print(f"\nwould run {len(bundles)} teacher call(s) "
              f"-> up to {planned} example(s) before quality and dedup")
        return 0

    teacher = select_teacher(
        gen.teacher,
        model=gen.model,
        timeout=gen.timeout_s,
        allow_metered_env=args.allow_metered_env,
    )
    if getattr(teacher, "subscription_only", False):
        print("[teacher] running on subscription credentials only "
              "(metered keys removed from the subprocess environment)")

    out_path = resolve_path(args.out or f"{cfg.generated_dir}/real-candidates.jsonl")
    examples, report = run_generation(
        docs_root, teacher, cfg,
        output_path=out_path,
        resume=args.resume,
        max_bundles=args.max_bundles,
        project_name=args.project_name,
    )

    if not args.no_dedup and examples:
        kept, dropped = deduplicate(examples, cfg.dedup)
        print(f"\n[dedup] {len(kept)} kept, {len(dropped)} dropped")
        if dropped:
            reasons: dict[str, int] = {}
            for row in dropped:
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"  {count:5d}  {reason}")
            dedup_path = out_path.with_name(out_path.stem + "-duplicates.jsonl")
            dedup_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in dropped) + "\n",
                encoding="utf-8")
        write_jsonl(out_path, kept)
        examples = kept

    manifest = {
        "docs_root": str(docs_root),
        "corpus": corpus_summary,
        "generation": gen.model_dump(),
        "report": report.as_dict(),
        "examples_final": len(examples),
        "output": str(out_path),
    }
    manifest_path = out_path.with_name(out_path.stem + "-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(examples)} example(s) -> {out_path}")
    print(f"wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
