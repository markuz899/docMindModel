"""Persistent cache for teacher requests.

Subscription quota is the scarcest resource in this pipeline, so an identical
request must never be sent twice -- across runs, across crashes, across days.
The key is a hash of everything that could change the answer: provider, CLI
version, model, prompt version, the documentation itself, and the generation
options. Change any of those and you get a new key, which is the point: a
cached answer is only reusable if it was produced under the same conditions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import resolve_path

# Bump when the teacher prompt changes meaning; it invalidates every entry.
# 3: rule 8 asks for an explanation with a shape instead of "be concise". v2
#    answers averaged 332 chars of extracted span, and that is what the model
#    learned to produce.
PROMPT_VERSION = "4"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes,
                "hit_rate": round(self.hit_rate, 4)}


@dataclass
class TeacherCache:
    root: Path = field(default_factory=lambda: resolve_path("data/cache/teacher"))
    enabled: bool = True
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def request_hash(
        provider: str,
        provider_version: str | None,
        model: str | None,
        system: str,
        user: str,
        schema: dict | None,
        options: dict | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "provider_version": provider_version,
                "model": model,
                "prompt_version": prompt_version,
                "system": system,
                "user": user,
                "schema": schema,
                "options": options or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        if not self.enabled:
            self.stats.misses += 1
            return None
        path = self._path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry.get("response")

    def put(self, key: str, response: str, meta: dict | None = None) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "response": response, "meta": meta or {},
                   "created_at": time.time()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a killed run never leaves a truncated entry
        self.stats.writes += 1

    def entries(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.exists() else 0


@dataclass
class RunState:
    """Resumable progress for one generation run.

    The cache already prevents re-asking the teacher, but the run state records
    which work units are finished so a resumed run does not even rebuild them,
    and so `--resume` can report honestly where it left off.
    """

    path: Path
    completed: set[str] = field(default_factory=set)
    counters: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @classmethod
    def load(cls, path: str | Path, resume: bool) -> "RunState":
        target = resolve_path(path)
        if resume and target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            return cls(
                path=target,
                completed=set(data.get("completed", [])),
                counters=data.get("counters", {}),
                started_at=data.get("started_at", time.time()),
            )
        return cls(path=target)

    def done(self, unit_id: str) -> bool:
        return unit_id in self.completed

    def mark(self, unit_id: str) -> None:
        self.completed.add(unit_id)

    def bump(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed": sorted(self.completed),
            "counters": self.counters,
            "started_at": self.started_at,
            "updated_at": time.time(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
