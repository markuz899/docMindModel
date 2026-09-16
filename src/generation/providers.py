"""Teacher-model providers.

The generation pipeline only needs ``complete(system, user) -> str``; keeping
that surface tiny is what makes the teacher swappable. API keys are optional
and read from the environment -- the mock provider keeps the pipeline runnable
and testable with no keys at all.
"""

from __future__ import annotations

import os
import re
from typing import Protocol


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class MockProvider:
    """Deterministic offline stand-in.

    It reads the documentation blocks back out of the prompt and emits one
    grounded example and one unanswerable example per call. Not a quality
    teacher -- a way to exercise the whole pipeline in CI without a network.
    """

    name = "mock"
    _BLOCK_RE = re.compile(r"\[(?P<source>[^\[\]\n]+?) — (?P<heading>[^\[\]\n]+?)\]\n(?P<body>.*?)(?=\n\[|\Z)", re.S)

    def __init__(self, model: str = "mock", **_: object) -> None:
        self.model = model

    def complete(self, system: str, user: str) -> str:
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
                "answerability": "full",
                "relevant_sources": [f"{first['source']}#{first['heading']}"],
                "must_include": [],
            },
            {
                "question": f"Quali metriche di latenza sono previste per {first['heading']}?",
                "answer": (
                    "La documentazione fornita descrive la sezione "
                    f"{first['heading']} ma non contiene informazioni sulle metriche "
                    "di latenza, quindi non è possibile rispondere con le fonti disponibili."
                ),
                "category": "how_it_works",
                "difficulty": "medium",
                "answerability": "none",
                "relevant_sources": [],
                "must_include": [],
            },
        ]
        return json.dumps(items, ensure_ascii=False)


class OpenAIProvider:
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

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:
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

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


_PROVIDERS = {"mock": MockProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider}


def get_provider(name: str, **kwargs: object) -> LLMProvider:
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; available: {sorted(_PROVIDERS)}") from None
    return factory(**kwargs)  # type: ignore[arg-type]
