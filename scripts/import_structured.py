#!/usr/bin/env python
"""Convert the RAG chat corpus into the project's Example schema.

    python scripts/import_structured.py

`data/generated/rag_chat_v3.jsonl` is raw chat turns with a [Dx] citation
style and its own system prompt; `data/structured_response.json` is a patch
file that replaces the assistant turn of the examples it lists. Neither can be
trained on directly -- src/prompt.py is the single prompt contract, and a
second citation format in the training data teaches the model two answers to
the same question.

So: apply the patches, parse the retrieved context back into chunks, rewrite
[D1] into the canonical [source - heading] token, and emit candidates that
`build_real_dataset.py` can read alongside the generated corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset_config
from src.dataset.io import read_jsonl, write_jsonl
from src.dataset.quality import check_example
from src.dataset.schema import Example
from src.prompt import citation

QUESTION_RE = re.compile(r"^(?:DOMANDA|QUESTION):\s*", re.M)
MD_HEAD_RE = re.compile(
    r"^---\s*\[(?P<id>D\d+)\]\s*(?P<source>.+?)\s*·\s*(?P<date>[\d-]+)\s*·.*?---\s*$", re.M
)
XML_DOC_RE = re.compile(
    r'<doc\s+id="(?P<id>[^"]+)"\s+source="(?P<source>[^"]+)"\s+date="(?P<date>[^"]+)"'
    r'[^>]*>(?P<content>.*?)</doc>', re.S
)


def split_question(user_message: str) -> tuple[str, str]:
    match = QUESTION_RE.search(user_message)
    if not match:
        raise ValueError("no DOMANDA/QUESTION marker")
    return user_message[: match.start()].strip(), user_message[match.end():].strip()


def parse_chunks(context_text: str, style: str) -> list[dict]:
    if style == "json":
        payload = context_text[context_text.index("["):]
        return [
            {"id": d["doc_id"], "source": clean_source(d["source"]), "date": d["date"],
             "content": d["content"]}
            for d in json.loads(payload)
        ]
    if style == "xml":
        return [
            {"id": m["id"], "source": clean_source(m["source"]), "date": m["date"],
             "content": m["content"].strip()}
            for m in XML_DOC_RE.finditer(context_text)
        ]
    if style == "md":
        heads = list(MD_HEAD_RE.finditer(context_text))
        return [
            {"id": h["id"], "source": clean_source(h["source"]), "date": h["date"],
             "content": context_text[h.end(): heads[i + 1].start() if i + 1 < len(heads)
                                     else len(context_text)].strip()}
            for i, h in enumerate(heads)
        ]
    raise ValueError(f"unknown context_style {style!r}")


def clean_source(source: str) -> str:
    """The citation token uses " - " as its separator, so a source that already
    contains one cannot be parsed back out of an answer."""
    return re.sub(r"\s*[-\u2013\u2014]\s*", " / ", source).strip()


# ponytail: doc id stays in the heading so source+heading is unique even when a
# corpus retrieves several chunks of the same document on different dates.
def heading_of(chunk: dict) -> str:
    return f"{chunk['id']} · {chunk['date']}"


def convert(record: dict, patches: dict[str, dict]) -> Example:
    meta = record["meta"]
    patch = patches.get(meta["id"], {})
    context_text, question = split_question(record["messages"][1]["content"])
    chunks = parse_chunks(context_text, meta["context_style"])
    if not chunks:
        raise ValueError("no chunks parsed")

    answer = patch.get("answer") or record["messages"][2]["content"]
    by_id = {c["id"]: citation(c["source"], heading_of(c)) for c in chunks}
    missing = {m for m in re.findall(r"\[(D\d+)\]", answer) if m not in by_id}
    if missing:
        raise ValueError(f"answer cites {sorted(missing)}, not in context")
    answer = re.sub(r"\[(D\d+)\]", lambda m: by_id[m.group(1)], answer)

    traits = meta.get("traits", [])
    category, answerable = "how_it_works", "full"
    if "conflict" in traits:
        category = "contradictory_context"
    elif "gap" in traits:
        category, answerable = "partially_answerable", "partial"
    elif len(chunks) > 2:
        category = "multi_source"

    return Example(
        id=meta["id"],
        question=question,
        context=[
            {"source": c["source"], "heading": heading_of(c), "content": c["content"]}
            for c in chunks
        ],
        answer=answer,
        category=category,
        answerable=answerable,
        difficulty="medium",
        project=f"rag-{meta['domain']}",
        relevant_sources=sorted(
            f"{c['source']}#{heading_of(c)}" for c in chunks
            if by_id[c["id"]] in answer
        ),
        tags=["structured-layout", *traits],
        origin="structured_response",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat", default="data/generated/rag_chat_v3.jsonl")
    parser.add_argument("--patches", default="data/structured_response.json")
    parser.add_argument("--out", default="artifacts/corpus/structured.jsonl")
    parser.add_argument("--config", default="configs/dataset.real.yaml")
    parser.add_argument("--append", metavar="SPLIT",
                        help="also append the kept examples to an existing split, "
                             "e.g. data/real/train/train.jsonl")
    args = parser.parse_args()

    patches = {
        e["id"]: e for e in json.loads(Path(args.patches).read_text(encoding="utf-8"))["examples"]
    }
    records = [json.loads(line) for line in Path(args.chat).read_text(encoding="utf-8").splitlines() if line.strip()]

    examples, failed = [], []
    for record in records:
        try:
            examples.append(convert(record, patches))
        except (ValueError, KeyError) as exc:
            failed.append((record.get("meta", {}).get("id", "?"), str(exc)))

    unused = sorted(set(patches) - {e.id for e in examples})
    print(f"converted {len(examples)}/{len(records)} ({len(patches)} patches, "
          f"{len(patches) - len(unused)} applied)")
    for example_id, reason in failed:
        print(f"  skipped {example_id}: {reason}")
    if unused:
        print(f"  patches with no matching chat record: {unused}")

    quality = load_dataset_config(args.config).quality
    kept, dropped = [], []
    for example in examples:
        report = check_example(example, quality)
        (kept if report.ok else dropped).append(example if report.ok else report)
    print(f"quality gate: {len(kept)} passed, {len(dropped)} rejected")
    for report in dropped:
        print(f"  {report.example_id}: {'; '.join(report.issues)}")

    write_jsonl(Path(args.out), kept)
    print(f"wrote {len(kept)} -> {args.out}")

    if args.append:
        split = Path(args.append)
        existing = read_jsonl(split)
        known = {e.id for e in existing}
        fresh = [e for e in kept if e.id not in known]
        write_jsonl(split, existing + fresh)
        print(f"appended {len(fresh)} to {split} ({len(existing) + len(fresh)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
