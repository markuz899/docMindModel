"""Teacher provider registry.

The generation pipeline only needs ``complete(system, user, schema) -> str``;
keeping that surface tiny is what makes the teacher swappable.

Default teachers are the local CLIs (`codex`, `claude`), which run against an
existing subscription. The metered API providers still exist for anyone who
wants them, but they are never selected automatically and never used as a
fallback -- see ``select_teacher``.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from src.generation.cli_teachers import (
    ClaudeCodeCliTeacher,
    CodexCliTeacher,
    HealthReport,
    TeacherHealth,
)

# Providers that cost money per token. Never chosen by `auto`, never a fallback.
METERED_PROVIDERS = frozenset({"openai", "anthropic"})


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict | None = None) -> str: ...

    def provider_name(self) -> str: ...

    def provider_version(self) -> str | None: ...

    def supports_structured_output(self) -> bool: ...

    def health_check(self) -> HealthReport: ...


class _SimpleProvider:
    """Default identity/health surface for providers that are just an API call."""

    name = "simple"

    def provider_name(self) -> str:
        return self.name

    def provider_version(self) -> str | None:
        return getattr(self, "model", None)

    def supports_structured_output(self) -> bool:
        return False

    def health_check(self) -> HealthReport:
        return HealthReport(self.name, TeacherHealth.AVAILABLE_AUTHENTICATED,
                            self.provider_version(), "no external check performed")


class MockProvider(_SimpleProvider):
    """Deterministic offline stand-in.

    It reads the documentation blocks back out of the prompt and emits one
    grounded example and one unanswerable example per call. Not a quality
    teacher -- a way to exercise the whole pipeline in CI without a network.
    """

    name = "mock"
    _BLOCK_RE = re.compile(r"\[(?P<source>[^\[\]\n]+?) — (?P<heading>[^\[\]\n]+?)\]\n(?P<body>.*?)(?=\n\[|\Z)", re.S)

    def __init__(self, model: str = "mock", **_: object) -> None:
        self.model = model

    def supports_structured_output(self) -> bool:
        return True

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        import json

        blocks = [m.groupdict() for m in self._BLOCK_RE.finditer(user)]
        if not blocks:
            return "[]"
        first = blocks[0]
        sentence = " ".join(first["body"].split())[:220].rstrip(". ") + "."
        cite = f"[{first['source']} — {first['heading']}]"
        items = [
            {
                "question": f"Come funziona {first['heading']}?",
                "answer": f"Secondo la documentazione: {sentence} {cite}",
                "category": "how_it_works",
                "difficulty": "easy",
                # both spellings on purpose: the parser accepts either
                "answerable": "full",
                "answerability": "full",
                "required_sources": [f"{first['source']}#{first['heading']}"],
                "relevant_sources": [f"{first['source']}#{first['heading']}"],
                "facts": [sentence[:80]],
                "unsupported_claims": [],
                "must_include": [],
            },
            {
                "question": f"Quali metriche di latenza sono previste per {first['heading']}?",
                "answer": (
                    "La documentazione fornita descrive la sezione "
                    f"{first['heading']} ma non contiene informazioni sulle metriche "
                    "di latenza, quindi non è possibile rispondere con le fonti disponibili."
                ),
                "category": "unanswerable",
                "difficulty": "medium",
                "answerable": "none",
                "answerability": "none",
                "required_sources": [],
                "relevant_sources": [],
                "facts": [],
                "unsupported_claims": ["latenza p99"],
                "must_include": [],
            },
        ]
        return json.dumps({"examples": items}, ensure_ascii=False)


class OpenAIProvider(_SimpleProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4,
                 max_output_tokens: int = 1500, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install openai to use the openai provider") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:  # pragma: no cover - network
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(_SimpleProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", temperature: float = 0.4,
                 max_output_tokens: int = 1500, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pip install anthropic to use the anthropic provider") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key)
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:  # pragma: no cover - network
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


_PROVIDERS = {
    "codex": CodexCliTeacher,
    "claude": ClaudeCodeCliTeacher,
    "mock": MockProvider,
    # metered, opt-in only
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

# Order tried by --teacher auto. Metered providers are absent on purpose.
AUTO_ORDER = ("codex", "claude")


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; available: {sorted(_PROVIDERS)}") from None
    return factory(**kwargs)  # type: ignore[arg-type]


def select_teacher(name: str, **kwargs: object) -> LLMProvider:
    """Resolve --teacher, including `auto`.

    `auto` tries the local CLIs in order and stops at the first authenticated
    one. It never reaches for a metered API: running out of subscription quota
    must fail loudly, not quietly start charging per token.
    """
    if name != "auto":
        if name in METERED_PROVIDERS:
            print(
                f"NOTE: '{name}' is a metered API provider and is billed per token. "
                "It was selected explicitly, so it will be used."
            )
        return get_provider(name, **kwargs)

    reports: list[HealthReport] = []
    for candidate in AUTO_ORDER:
        provider = get_provider(candidate, **kwargs)
        report = provider.health_check()
        reports.append(report)
        if report.usable and report.metered_blocking and not kwargs.get("allow_metered_env"):
            print(f"[teacher] skipping {candidate}: would be billed per token")
            for warning in report.metered_blocking:
                print(f"[teacher]   {warning}")
            continue
        if report.usable:
            print(f"[teacher] auto-selected {candidate} ({report.version or 'unknown version'})")
            for warning in report.metered_blocking:
                print(f"[teacher] BILLING RISK: {warning}")
            for warning in report.metered_warnings:
                print(f"[teacher] note: {warning}")
            return provider

    detail = "\n".join(r.render() for r in reports)
    raise SystemExit(
        "No local teacher CLI is usable on a subscription.\n\n"
        f"{detail}\n\n"
        "Install and log in to one of them:\n"
        "  codex login      (https://github.com/openai/codex)\n"
        "  claude auth login\n\n"
        "There is deliberately no automatic fallback to a metered API. If you do\n"
        "want to pay per token, pass --teacher openai or --teacher anthropic."
    )


def health_report_all(**kwargs: object) -> list[HealthReport]:
    """Health of every local CLI teacher, for `--health-check`."""
    return [get_provider(name, **kwargs).health_check() for name in AUTO_ORDER]
