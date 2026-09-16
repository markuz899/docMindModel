"""JSONL read/write. Tolerant on read (reports bad lines), strict on write."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import ValidationError

from src.config import resolve_path
from src.dataset.schema import Example


def read_jsonl_raw(path: str | Path) -> Iterator[dict]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON ({exc})") from exc


def read_jsonl(path: str | Path, strict: bool = True) -> list[Example]:
    examples, errors = [], []
    for i, obj in enumerate(read_jsonl_raw(path), 1):
        try:
            examples.append(Example(**obj))
        except ValidationError as exc:
            errors.append(f"line {i} (id={obj.get('id')}): {exc.error_count()} error(s)")
            if strict:
                raise ValueError(f"{path}:{i}: {exc}") from exc
    if errors:
        print(f"[io] skipped {len(errors)} invalid example(s) in {path}")
    return examples


def write_jsonl(path: str | Path, examples: Iterable[Example | dict]) -> int:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            payload = ex.to_dict() if isinstance(ex, Example) else ex
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count
