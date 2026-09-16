"""Markdown -> heading-scoped chunks, the unit both the retriever and the
dataset generator work with."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import ChunkingConfig, resolve_path
from src.dataset.schema import ContextChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def split_markdown(text: str, source: str, cfg: ChunkingConfig | None = None) -> list[ContextChunk]:
    """Split on headings, ignoring '#' inside fenced code blocks."""
    cfg = cfg or ChunkingConfig()
    chunks: list[ContextChunk] = []
    heading = "Overview"
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush(current_heading: str) -> None:
        content = "\n".join(buffer).strip()
        if len(content) < cfg.min_chunk_chars:
            return
        chunks.append(
            ContextChunk(
                source=source,
                heading=current_heading,
                content=content[: cfg.max_chunk_chars],
            )
        )

    for line in text.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence = False
            buffer.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            if cfg.min_heading_level <= level <= cfg.max_heading_level:
                flush(heading)
                heading = match.group(2).strip() or "Overview"
                buffer = []
                continue
        buffer.append(line)

    flush(heading)
    return chunks


def load_markdown_file(path: str | Path, cfg: ChunkingConfig | None = None) -> list[ContextChunk]:
    p = resolve_path(path)
    return split_markdown(p.read_text(encoding="utf-8"), p.name, cfg)


def load_corpus(
    root: str | Path, cfg: ChunkingConfig | None = None
) -> dict[str, list[ContextChunk]]:
    """Load ``root/<project>/**/*.md`` into {project: chunks}.

    Markdown sitting directly in ``root`` is grouped under the root's own name.
    """
    base = resolve_path(root)
    corpus: dict[str, list[ContextChunk]] = {}
    if not base.exists():
        return corpus
    for path in sorted(base.rglob("*.md")):
        rel = path.relative_to(base)
        project = rel.parts[0] if len(rel.parts) > 1 else base.name
        corpus.setdefault(project, []).extend(load_markdown_file(path, cfg))
    return {k: v for k, v in corpus.items() if v}
