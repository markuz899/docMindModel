#!/usr/bin/env python
"""Assemble several documentation sources into one multi-project corpus.

    python scripts/stage_corpus.py --out artifacts/corpus/real \
        ~/code/service-a/docs ~/code/service-b/docs

Why this exists: the split policy holds out *whole projects*, so a corpus with
one project cannot produce an honest test set. This stages each source as its
own top-level directory, which is exactly what `load_corpus` groups on.

It copies documentation files only, never source code, and writes nothing back
to the sources. The default destination is under artifacts/, which is
gitignored -- private documentation does not land in this repository unless you
move it there deliberately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path
from src.generation.markdown import DOC_SUFFIXES, iter_doc_files, load_corpus


def project_label(source: Path) -> str:
    """`~/code/service-a/docs` -> `service-a`, not a directory called `docs`."""
    name = source.name
    if name.lower() in {"docs", "doc", "documentation", "wiki"} and source.parent.name:
        name = source.parent.name
    return name.replace(" ", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="documentation directories")
    parser.add_argument("--out", default="artifacts/corpus/real")
    parser.add_argument("--min-sections", type=int, default=5,
                        help="skip sources thinner than this")
    parser.add_argument("--clean", action="store_true", help="wipe --out first")
    args = parser.parse_args()

    out_root = resolve_path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    staged, skipped = {}, []
    for raw in args.sources:
        source = Path(raw).expanduser().resolve()
        if not source.is_dir():
            skipped.append((raw, "not a directory"))
            continue
        label = project_label(source)
        files = list(iter_doc_files(source))
        if not files:
            skipped.append((raw, f"no {'/'.join(DOC_SUFFIXES)} files"))
            continue

        target = out_root / label
        target.mkdir(parents=True, exist_ok=True)
        for path in files:
            rel = path.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)

        sections = sum(len(v) for v in load_corpus(target, project_name=label).values())
        if sections < args.min_sections:
            shutil.rmtree(target)
            skipped.append((raw, f"only {sections} sections"))
            continue
        staged[label] = {"source": str(source), "files": len(files), "sections": sections}

    if not staged:
        raise SystemExit("nothing staged; every source was skipped")

    print(f"{'project':<32}{'files':>7}{'sections':>10}")
    print("-" * 49)
    for label, info in sorted(staged.items()):
        print(f"{label:<32}{info['files']:>7}{info['sections']:>10}")
    total = sum(i["sections"] for i in staged.values())
    print("-" * 49)
    print(f"{'total':<32}{sum(i['files'] for i in staged.values()):>7}{total:>10}")
    for raw, reason in skipped:
        print(f"skipped {raw}: {reason}")

    manifest = {"staged": staged, "skipped": skipped, "out": str(out_root),
                "total_sections": total}
    (out_root / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nstaged {len(staged)} project(s) -> {out_root}")
    print(f"next: python -m src.generation.generate --docs {args.out} --teacher auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
