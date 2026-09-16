"""Markdown -> heading-scoped chunks, the unit both the retriever and the
dataset generator work with."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.config import ChunkingConfig, resolve_path
from src.dataset.schema import ContextChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


# Directories that are never documentation.
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "dist", "build", ".next", "out",
    "coverage", "venv", ".venv", "env", "__pycache__", ".pytest_cache", "target",
    ".idea", ".vscode", ".cache", ".turbo", "vendor", "site-packages",
    ".mypy_cache", ".ruff_cache", ".tox", "bower_components", ".gradle",
})
DOC_SUFFIXES = (".md", ".mdx", ".txt")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_markdown(
    text: str,
    source: str,
    cfg: ChunkingConfig | None = None,
    relative_path: str | None = None,
) -> list[ContextChunk]:
    """Split on headings, ignoring '#' inside fenced code blocks."""
    cfg = cfg or ChunkingConfig()
    chunks: list[ContextChunk] = []
    heading = "Overview"
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush(current_heading: str, ancestry: list[str]) -> None:
        content = "\n".join(buffer).strip()
        if len(content) < cfg.min_chunk_chars:
            return
        body = content[: cfg.max_chunk_chars]
        chunks.append(
            ContextChunk(
                source=source,
                heading=current_heading,
                content=body,
                relative_path=relative_path,
                heading_path=ancestry or None,
                content_hash=content_hash(body),
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
                flush(heading, [h for _, h in heading_stack])
                heading = match.group(2).strip() or "Overview"
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                buffer = []
                continue
        buffer.append(line)

    flush(heading, [h for _, h in heading_stack])
    return chunks


def split_plain_text(
    text: str, source: str, cfg: ChunkingConfig | None = None,
    relative_path: str | None = None,
) -> list[ContextChunk]:
    """A .txt file has no headings: keep it whole, named after the file."""
    cfg = cfg or ChunkingConfig()
    body = text.strip()[: cfg.max_chunk_chars]
    if len(body) < cfg.min_chunk_chars:
        return []
    return [ContextChunk(source=source, heading=Path(source).stem, content=body,
                         relative_path=relative_path, content_hash=content_hash(body))]


def load_markdown_file(
    path: str | Path, cfg: ChunkingConfig | None = None, relative_path: str | None = None
) -> list[ContextChunk]:
    p = resolve_path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    splitter = split_plain_text if p.suffix.lower() == ".txt" else split_markdown
    return splitter(text, p.name, cfg, relative_path or p.name)


def iter_doc_files(root: Path, suffixes: tuple[str, ...] = DOC_SUFFIXES):
    """Walk a documentation tree, skipping build output and dependency trees."""
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes or not path.is_file():
            continue
        if any(part in IGNORED_DIRS or part.startswith(".") and part not in (".",)
               for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def load_corpus(
    root: str | Path,
    cfg: ChunkingConfig | None = None,
    suffixes: tuple[str, ...] = DOC_SUFFIXES,
    project_name: str | None = None,
) -> dict[str, list[ContextChunk]]:
    """Load a documentation tree into {project: chunks}.

    Each immediate subdirectory of ``root`` is one project -- the grouping the
    split policy needs to hold a whole corpus out of training. Files sitting
    directly in ``root`` are grouped under ``project_name`` or the root's name.
    """
    base = resolve_path(root)
    corpus: dict[str, list[ContextChunk]] = {}
    if not base.exists():
        return corpus
    for path in iter_doc_files(base, suffixes):
        rel = path.relative_to(base)
        project = rel.parts[0] if len(rel.parts) > 1 else (project_name or base.name)
        corpus.setdefault(project, []).extend(
            load_markdown_file(path, cfg, relative_path=str(rel))
        )
    return {k: v for k, v in corpus.items() if v}
