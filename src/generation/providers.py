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
    TeacherCallError,
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


class SelfHostedProvider(_SimpleProvider):
    """A self-hosted OpenAI-compatible endpoint (e.g. a local open-weight model).

    Free to call -- not a metered API and not a subscription CLI -- so it is
    never part of `--teacher auto` and must be selected explicitly with
    `--teacher selfhosted`. Credentials and endpoint come from the environment
    only, never hardcoded here:

      DOCMIND_SELFHOSTED_URL   e.g. https://host/v1/chat/completions
      DOCMIND_SELFHOSTED_MODEL e.g. qwythos-9b
      DOCMIND_SELFHOSTED_AUTH  "user:password" for HTTP Basic auth (optional)

    Weaker models need more hand-holding than the prompt in prompts.py gives
    codex/claude, or they default to refusing everything and reciting the
    QUESTION STYLE examples verbatim instead of writing new questions.
    _ENRICHMENT below fixes both, measured against this project's own quality
    gate (src/dataset/quality.py) on real bundles before this was wired in:
    without it, 0/5 examples passed (no inline citation, wrong answerability);
    with it, 10/10 passed across a plain bundle and a distractor+refusal one.
    """

    name = "selfhosted"

    _ENRICHMENT = """

IMPORTANT, READ CAREFULLY:
* CITATION FORMAT -- THE SINGLE MOST IMPORTANT RULE, CHECKED MECHANICALLY:
  every non-refusal `answer` string MUST contain the literal token
  "[file — heading]" (with the em-dash "—", copied verbatim from the block
  header) INSIDE the answer text, at the point where that fact is used. This
  applies no matter how short or how long the answer is -- a one-sentence
  answer needs the bracket exactly as much as a five-paragraph one. Listing
  the source in `required_sources` is NOT enough by itself and does not
  substitute for it. Every single example below has one; match that, every
  time. A "none" refusal has no citation and that is correct -- do not invent
  one for a refusal.
* The example questions shown above in QUESTION STYLE are STYLE illustrations
  from an unrelated fictitious project. Reusing any of them, verbatim or
  reworded, is a defect. Every question must be about entities/files/behaviour
  that actually appear in the documentation blocks below.
* Follow the answerable/category plan for each numbered item EXACTLY. If an
  item says answerable=full, the blocks below DO contain the fact -- find it
  and answer it, don't default to a refusal. If an item says answerable=none,
  the blocks do NOT contain it -- refuse and say what would be needed, don't
  guess.
* The WORKED EXAMPLES below are about a DIFFERENT, fictitious project
  ("billing.md", webhooks). They show the shape of a good answer only.
  Copying their questions, answers, or billing/webhook content -- verbatim or
  reworded -- into your own output is a defect, exactly like reusing a
  QUESTION STYLE example. Every example you write must be built from the
  actual documentation blocks given to you below, never from these.

WORKED EXAMPLES (fictitious project, for format only -- never copy their content):
  Blocks given:
    [billing.md — Retry policy]
    Failed webhook deliveries are retried 3 times with exponential backoff
    starting at 30s, then marked dead-lettered.
  Plan item 1: answerable=full, category=configuration -- a short direct answer:
    {
      "question": "quante volte viene ritentata una webhook fallita?",
      "answer": "Una webhook fallita viene ritentata 3 volte con backoff "
        "esponenziale a partire da 30s, poi finisce in dead-letter "
        "[billing.md — Retry policy].",
      "category": "configuration",
      "difficulty": "easy",
      "answerable": "full",
      "required_sources": ["billing.md#Retry policy"],
      "facts": ["retried 3 times", "backoff starts at 30s", "dead-lettered after retries"],
      "unsupported_claims": []
    }
  Plan item 2: answerable=full, category=how_it_works -- a longer, explanatory
  answer (the ANSWER SHAPE rule); notice the bracket is still there, inside
  the explanation, not just tacked on at the end:
    {
      "question": "perché una webhook fallita non viene ritentata all'infinito?",
      "answer": "Il retry è limitato a 3 tentativi con backoff esponenziale "
        "a partire da 30s [billing.md — Retry policy]. Il backoff esiste per "
        "non sommergere l'endpoint di destinazione con richieste ravvicinate "
        "quando sta già fallendo; il limite di 3 tentativi esiste perché "
        "oltre quella soglia un fallimento ripetuto è quasi certamente "
        "permanente (endpoint spento, credenziali scadute), non transitorio, "
        "quindi continuare a ritentare sprecherebbe solo risorse. Per chi "
        "integra una webhook, questo significa che dopo il terzo fallimento "
        "il messaggio finisce in dead-letter [billing.md — Retry policy] e "
        "va gestito manualmente, non aspettato.",
      "category": "how_it_works",
      "difficulty": "medium",
      "answerable": "full",
      "required_sources": ["billing.md#Retry policy"],
      "facts": ["max 3 retries", "exponential backoff from 30s", "dead-lettered after limit", "manual handling needed after dead-letter"],
      "unsupported_claims": []
    }
"""

    def __init__(self, model: str | None = None, temperature: float = 0.4,
                 timeout: int = 300, **_: object) -> None:
        self.url = os.environ.get("DOCMIND_SELFHOSTED_URL")
        self.model = model or os.environ.get("DOCMIND_SELFHOSTED_MODEL")
        if not self.url or not self.model:
            raise RuntimeError(
                "the selfhosted teacher needs DOCMIND_SELFHOSTED_URL and "
                "DOCMIND_SELFHOSTED_MODEL set in the environment "
                "(DOCMIND_SELFHOSTED_AUTH too, if the endpoint needs HTTP Basic auth)."
            )
        self.temperature = temperature
        self.timeout = timeout or 300
        self._auth_header = None
        auth = os.environ.get("DOCMIND_SELFHOSTED_AUTH")
        if auth:
            import base64

            self._auth_header = "Basic " + base64.b64encode(auth.encode()).decode()

    def supports_structured_output(self) -> bool:
        return True

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        import json
        import urllib.error
        import urllib.request

        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system + self._ENRICHMENT},
                {"role": "user", "content": user},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": self.temperature,
        }
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "examples", "schema": schema, "strict": True},
            }
        headers = {"Content-Type": "application/json"}
        if self._auth_header:
            headers["Authorization"] = self._auth_header
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST", headers=headers
        )
        # process() in pipeline.py only catches TeacherCallError / TeacherUsageLimit
        # and documents itself as "never raises" -- a bare network exception here
        # would crash the whole run instead of counting as one failed bundle.
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except TimeoutError as exc:
            raise TeacherCallError(f"selfhosted request timed out after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            raise TeacherCallError(f"selfhosted request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TeacherCallError(f"selfhosted returned invalid JSON: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherCallError(
                f"selfhosted returned an unexpected response shape: {exc}"
            ) from exc


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
    # free, but not a subscription CLI -- opt-in only, never part of `auto`
    "selfhosted": SelfHostedProvider,
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
