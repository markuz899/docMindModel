#!/usr/bin/env python
"""Ask the model a question against a documentation context.

    python -m src.inference.chat --context docs/ --question "Come funziona GET /me?"
    python -m src.inference.chat --context context.json          # interactive
    python -m src.inference.chat --adapter artifacts/adapter --context docs/

--context accepts a Markdown file, a directory of Markdown, a JSON list of
{source, heading, content} chunks, or a JSONL dataset file (uses the first row's
context). This mirrors what the DocMind desktop app hands over at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import resolve_path
from src.dataset.schema import ContextChunk
from src.generation.markdown import load_markdown_file
from src.inference.engine import DocMindEngine
from src.prompt import SYSTEM_PROMPT


def load_context(spec: str) -> list[ContextChunk]:
    path = resolve_path(spec)
    if not path.exists():
        raise SystemExit(f"context not found: {path}")
    if path.is_dir():
        chunks: list[ContextChunk] = []
        for md in sorted(path.rglob("*.md")):
            chunks.extend(load_markdown_file(md))
        if not chunks:
            raise SystemExit(f"no .md files under {path}")
        return chunks
    if path.suffix == ".md":
        return load_markdown_file(path)
    if path.suffix == ".jsonl":
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        return [ContextChunk(**c) for c in first["context"]]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("context", [])
        return [ContextChunk(**c) for c in data]
    raise SystemExit(f"unsupported context file type: {path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    parser.add_argument("--context", required=True)
    parser.add_argument("--question", default=None, help="omit for an interactive prompt")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=8,
                        help="cap the number of context chunks passed to the model")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()

    chunks = load_context(args.context)[: args.max_sources]
    print(f"context: {len(chunks)} section(s)")
    for c in chunks:
        print(f"  - [{c.source} — {c.heading}]")

    engine = DocMindEngine.load(args.model_config, args.adapter)
    print(f"model: {engine.label} on {engine.device}\n")

    payload = [c.model_dump() for c in chunks]

    def answer(question: str) -> None:
        if args.show_prompt:
            print("---- prompt ----")
            print(engine.render_prompt(question, payload, SYSTEM_PROMPT))
            print("----------------")
        print(engine.generate(question, payload, max_new_tokens=args.max_new_tokens))

    if args.question:
        answer(args.question)
        return 0

    print("Interactive mode. Empty line or Ctrl-D to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        print()
        answer(question)


if __name__ == "__main__":
    raise SystemExit(main())
