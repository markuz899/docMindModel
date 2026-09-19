"""The JSONL training example schema.

One line of the dataset = one (question, retrieved context, grounded answer)
triple, plus the metadata the evaluators need to score it.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    """Question shapes the model has to handle.

    The first block is the current vocabulary. The legacy block is kept so the
    v1 demo dataset still validates unchanged -- `route`/`api`,
    `dependencies`/`dependency`, `integrations`/`integration` and
    `errors`/`error_handling` are the same idea under two spellings, and
    rewriting the shipped v1 data to unify them would invalidate a published
    experiment for no benefit.
    """

    HOW_IT_WORKS = "how_it_works"
    API = "api"
    SERVICE = "service"
    DEPENDENCY = "dependency"
    ARCHITECTURE = "architecture"
    DATABASE = "database"
    DATA_FLOW = "data_flow"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    ERROR_HANDLING = "error_handling"
    BUG_INVESTIGATION = "bug_investigation"
    MULTI_SOURCE = "multi_source"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    UNANSWERABLE = "unanswerable"
    CONTRADICTORY_CONTEXT = "contradictory_context"
    EXTENSION_GUIDANCE = "extension_guidance"

    # --- legacy spellings, v1 demo dataset ---
    ROUTE = "route"
    DEPENDENCIES = "dependencies"
    INTEGRATIONS = "integrations"
    ERRORS = "errors"
    COMPARISON = "comparison"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Answerability(str, Enum):
    """How much of the question the supplied context actually supports."""

    FULL = "full"        # context answers the question
    PARTIAL = "partial"  # context answers part of it, the rest must be flagged unknown
    NONE = "none"        # context cannot answer it -> the answer must refuse


class ContextChunk(BaseModel):
    """One retrieved documentation section, as the retriever would hand it over."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description="File name, e.g. 04-api-reference.md")
    heading: str = Field(min_length=1, description="Section heading, e.g. GET /me")
    content: str = Field(min_length=1)
    # Optional retrieval metadata; ignored by training, used by the noise tests.
    relevant: bool | None = None
    # Provenance for real corpora. None (not []) when unset, so existing
    # datasets serialise byte-identically.
    relative_path: str | None = None
    heading_path: list[str] | None = Field(
        default=None, description="Heading ancestry, outermost first"
    )
    content_hash: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.source.strip(), self.heading.strip())

    def citation(self) -> str:
        from src.prompt import citation

        return citation(self.source, self.heading)


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    context: list[ContextChunk] = Field(min_length=1)
    answer: str = Field(min_length=1)

    # --- optional metadata (spec) ---
    category: Category | None = None
    difficulty: Difficulty | None = None
    contains_code: bool | None = None
    answerable: Answerability | None = None
    source_count: int | None = None

    # --- extensions used by the split policy and the evaluators ---
    project: str | None = Field(default=None, description="Group key for the split")
    relevant_sources: list[str] = Field(
        default_factory=list,
        description="Gold citations '<source>#<heading>' the answer should rely on",
    )
    must_include: list[str] = Field(
        default_factory=list, description="Key facts a correct answer has to mention"
    )
    must_not_include: list[str] = Field(
        default_factory=list, description="Facts absent from context; mentioning them is a hallucination"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Atomic claims the answer makes, each supposed to be in the context",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims the teacher itself flagged as not supported by the context",
    )
    teacher_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Audit trail: provider, CLI version, prompt version, request hash, timestamp",
    )
    tags: list[str] = Field(default_factory=list)  # e.g. distractor, contradiction, seed
    origin: str | None = None  # seed | teacher:<model> | synthetic

    @field_validator("answer", "question")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _fill_derived(self) -> "Example":
        if self.source_count is None:
            object.__setattr__(self, "source_count", len(self.context))
        if self.contains_code is None:
            object.__setattr__(self, "contains_code", "`" in self.answer or "```" in self.answer)
        return self

    # --- helpers -------------------------------------------------------

    def context_keys(self) -> set[tuple[str, str]]:
        return {c.key for c in self.context}

    def context_text(self) -> str:
        """Everything the model can legitimately draw on, citations included."""
        return "\n\n".join(
            f"{c.citation()}\n{c.heading}\n{c.content}" for c in self.context
        )

    # `relevant_sources` is the canonical name for what the teacher schema calls
    # `required_sources`; the parser maps one onto the other rather than storing
    # the same list twice.
    def relevant_keys(self) -> set[tuple[str, str]]:
        """Gold sources; falls back to chunks explicitly flagged relevant."""
        keys = set()
        for ref in self.relevant_sources:
            source, _, heading = ref.partition("#")
            keys.add((source.strip(), heading.strip()))
        if not keys:
            keys = {c.key for c in self.context if c.relevant is not False}
        return keys

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, mode="json")


def make_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"
