"""Teacher providers that drive an already-authenticated local coding CLI.

Why CLIs instead of the metered APIs: generating a few thousand training
examples through pay-per-token endpoints is expensive, while `codex` and
`claude` are already authenticated against a subscription. These providers
invoke them as plain subprocesses. They never read, copy or manipulate
credentials -- authentication is the user's business, done outside this repo.

Flags here were read from the installed CLIs (`codex exec --help`,
`claude --help`) rather than recalled; see docs/teacher-cli.md for the exact
versions this was verified against.

Two rules that are not negotiable:
  * the teacher runs read-only, with no tools and no access to this repository;
  * a CLI failure NEVER falls back to a metered API.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TeacherHealth(str, Enum):
    AVAILABLE_AUTHENTICATED = "available_authenticated"
    AVAILABLE_NOT_AUTHENTICATED = "available_not_authenticated"
    NOT_INSTALLED = "not_installed"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class HealthReport:
    provider: str
    status: TeacherHealth
    version: str | None = None
    detail: str = ""
    auth_method: str | None = None
    # Advisory: worth saying, but the auth probe shows the subscription is in use.
    metered_warnings: list[str] = field(default_factory=list)
    # Blocking: this configuration really would be billed per token.
    metered_blocking: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status is TeacherHealth.AVAILABLE_AUTHENTICATED

    def render(self) -> str:
        lines = [f"{self.provider}: {self.status.value}"]
        if self.version:
            lines[0] += f" (v{self.version})"
        if self.auth_method:
            lines[0] += f" via {self.auth_method}"
        if self.detail:
            lines.append(f"  {self.detail}")
        for warning in self.metered_blocking:
            lines.append(f"  BILLING RISK: {warning}")
        for warning in self.metered_warnings:
            lines.append(f"  note: {warning}")
        return "\n".join(lines)


class TeacherUsageLimit(RuntimeError):
    """The CLI reported a subscription/rate limit. Checkpoint and stop."""


class TeacherCallError(RuntimeError):
    """The CLI failed for a reason that is not a usage limit."""


# Substrings that mean "you are out of quota", not "this prompt failed".
# Deliberately specific: `codex exec` echoes the whole prompt to stdout, and
# documentation about API rate limiting would otherwise read as a quota error.
_USAGE_LIMIT_PATTERNS = (
    "usage limit", "rate limit exceeded", "rate_limit_error", "quota exceeded",
    "out of quota", "too many requests", "http 429", "status 429",
    "limit reached", "limit exceeded", "resets at", "try again later",
    "overloaded_error", "insufficient_quota", "you have hit your",
    "upgrade to continue", "plan limit",
)

# Only the last slice of each stream is inspected. `codex exec` echoes the whole
# prompt -- including the user's documentation -- before it reports anything, so
# scanning the full output would read "applies rate limits" in someone's gateway
# docs as a quota error.
_STREAM_TAIL_CHARS = 2000


def looks_like_usage_limit(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _USAGE_LIMIT_PATTERNS)


def failure_text(proc: subprocess.CompletedProcess) -> str:
    """Error surface of a failed CLI run, without the echoed prompt."""
    stderr_tail = (proc.stderr or "")[-_STREAM_TAIL_CHARS:].strip()
    stdout_tail = (proc.stdout or "")[-_STREAM_TAIL_CHARS:].strip()
    return f"{stderr_tail}\n{stdout_tail}".strip()


def metered_env_warnings(provider: str) -> list[str]:
    """Environment that could silently turn a subscription call into a billed one."""
    warnings: list[str] = []
    if provider == "claude":
        if os.environ.get("ANTHROPIC_API_KEY"):
            warnings.append(
                "ANTHROPIC_API_KEY is set. Claude Code may bill this key per token "
                "instead of using your subscription. Unset it in this shell "
                "(`unset ANTHROPIC_API_KEY`) if you want subscription usage."
            )
        for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
                    "CLAUDE_CODE_USE_VERTEX"):
            if os.environ.get(var):
                warnings.append(f"{var} is set; requests may be routed to a billed endpoint.")
    if provider == "codex":
        if os.environ.get("OPENAI_API_KEY"):
            warnings.append(
                "OPENAI_API_KEY is set. Codex prefers the ChatGPT login when present, "
                "but verify with `codex login status` that it says 'Logged in using ChatGPT'."
            )
    return warnings


def _run(argv: list[str], stdin: str | None = None, timeout: int = 300,
         cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a CLI without a shell. Argument list only -- never string concatenation."""
    try:
        return subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError as exc:
        raise TeacherCallError(f"{argv[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise TeacherCallError(f"{argv[0]} timed out after {timeout}s") from exc
    except KeyboardInterrupt:
        raise


class _CliTeacher:
    """Shared plumbing: version probe, health cache, structured-output contract."""

    binary = ""
    name = ""

    def __init__(self, model: str | None = None, timeout: int = 300,
                 allow_metered_env: bool = False, **_: object) -> None:
        self.model = model
        self.timeout = timeout
        self.allow_metered_env = allow_metered_env
        self._health: HealthReport | None = None

    # -- identity -------------------------------------------------------

    def provider_name(self) -> str:
        return self.name

    def provider_version(self) -> str | None:
        path = shutil.which(self.binary)
        if not path:
            return None
        try:
            proc = _run([self.binary, "--version"], timeout=30)
        except TeacherCallError:
            return None
        return (proc.stdout or proc.stderr).strip().splitlines()[0] if proc.returncode == 0 else None

    def supports_structured_output(self) -> bool:
        return True

    def health_check(self, refresh: bool = False) -> HealthReport:
        if self._health is None or refresh:
            self._health = self._probe_health()
        return self._health

    def _probe_health(self) -> HealthReport:  # pragma: no cover - overridden
        raise NotImplementedError

    # -- generation -----------------------------------------------------

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        health = self.health_check()
        if not health.usable:
            raise TeacherCallError(
                f"{self.name} is not usable: {health.status.value}. {health.detail}"
            )
        if health.metered_blocking and not self.allow_metered_env:
            raise TeacherCallError(
                f"{self.name}: refusing to run, this configuration is billed per token:\n  - "
                + "\n  - ".join(health.metered_blocking)
                + "\nFix the environment, or pass --allow-metered-env to accept the cost."
            )
        return self._generate(system, user, schema)

    def _generate(self, system: str, user: str, schema: dict | None) -> str:  # pragma: no cover
        raise NotImplementedError


class CodexCliTeacher(_CliTeacher):
    """OpenAI Codex CLI in non-interactive, read-only, sandboxed mode.

    Verified against codex-cli 0.154.0:
      codex exec --sandbox read-only --skip-git-repo-check --ephemeral
                 --output-schema FILE --output-last-message FILE -
    """

    binary = "codex"
    name = "codex"

    def _probe_health(self) -> HealthReport:
        version = self.provider_version()
        if not shutil.which(self.binary):
            return HealthReport(self.name, TeacherHealth.NOT_INSTALLED,
                                detail="`codex` is not on PATH")
        try:
            proc = _run([self.binary, "login", "status"], timeout=45)
        except TeacherCallError as exc:
            return HealthReport(self.name, TeacherHealth.UNKNOWN_ERROR, version, str(exc))
        output = (proc.stdout + proc.stderr).strip()
        warnings = metered_env_warnings("codex")
        if proc.returncode != 0 or "not logged in" in output.lower():
            return HealthReport(self.name, TeacherHealth.AVAILABLE_NOT_AUTHENTICATED, version,
                                f"run `codex login` ({output[:120]})", metered_warnings=warnings)
        lowered = output.lower()
        auth = "ChatGPT" if "chatgpt" in lowered else ("API key" if "api key" in lowered else None)
        blocking = []
        if auth == "API key":
            blocking.append(
                "codex is logged in with an API key, so every call is billed per token. "
                "Run `codex login` to use your ChatGPT subscription instead."
            )
        return HealthReport(self.name, TeacherHealth.AVAILABLE_AUTHENTICATED, version,
                            output[:160], auth_method=auth,
                            metered_warnings=warnings, metered_blocking=blocking)

    def _generate(self, system: str, user: str, schema: dict | None) -> str:
        # An empty scratch directory is the agent's whole world: it cannot read
        # this repository, and nothing it does can touch it.
        with tempfile.TemporaryDirectory(prefix="docmind-codex-") as workdir:
            out_file = Path(workdir) / "last_message.txt"
            argv = [
                self.binary, "exec",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color", "never",
                "--cd", workdir,
                "--output-last-message", str(out_file),
            ]
            if self.model:
                argv += ["--model", self.model]
            if schema:
                schema_file = Path(workdir) / "schema.json"
                schema_file.write_text(json.dumps(schema), encoding="utf-8")
                argv += ["--output-schema", str(schema_file)]
            argv.append("-")  # read the prompt from stdin

            prompt = f"{system}\n\n---\n\n{user}"
            proc = _run(argv, stdin=prompt, timeout=self.timeout, cwd=workdir)
            if proc.returncode != 0:
                # A quota problem is a *failure*; a success that merely mentions
                # rate limiting is documentation, not an error.
                detail = failure_text(proc)
                if looks_like_usage_limit(detail):
                    raise TeacherUsageLimit(f"codex usage limit: {detail[:400]}")
                raise TeacherCallError(
                    f"codex exec failed (exit {proc.returncode}): {detail[:600]}"
                )
            if out_file.exists() and out_file.read_text(encoding="utf-8").strip():
                return out_file.read_text(encoding="utf-8")
            return proc.stdout or ""


class ClaudeCodeCliTeacher(_CliTeacher):
    """Claude Code CLI in print mode with every tool disabled.

    Verified against claude 2.1.273:
      claude -p --tools "" --output-format json --json-schema JSON
             --system-prompt TEXT --strict-mcp-config

    `--bare` is deliberately NOT used: it forces ANTHROPIC_API_KEY / apiKeyHelper
    authentication and would bill per token instead of using the subscription.
    """

    binary = "claude"
    name = "claude"

    def _probe_health(self) -> HealthReport:
        version = self.provider_version()
        if not shutil.which(self.binary):
            return HealthReport(self.name, TeacherHealth.NOT_INSTALLED,
                                detail="`claude` is not on PATH")
        try:
            proc = _run([self.binary, "auth", "status", "--json"], timeout=45)
        except TeacherCallError as exc:
            return HealthReport(self.name, TeacherHealth.UNKNOWN_ERROR, version, str(exc))
        warnings = metered_env_warnings("claude")
        try:
            status = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return HealthReport(self.name, TeacherHealth.UNKNOWN_ERROR, version,
                                f"unparseable auth status: {(proc.stdout or proc.stderr)[:160]}",
                                metered_warnings=warnings)
        if not status.get("loggedIn"):
            return HealthReport(self.name, TeacherHealth.AVAILABLE_NOT_AUTHENTICATED, version,
                                "run `claude auth login`", metered_warnings=warnings)
        auth_method = status.get("authMethod")
        key_source = status.get("apiKeySource")
        blocking = []
        if key_source and key_source != "none":
            blocking.append(
                f"claude reports apiKeySource={key_source}: that credential is billed per "
                f"token, even though authMethod={auth_method}. Run `unset {key_source}` in "
                "this shell to use the subscription instead."
            )
        return HealthReport(self.name, TeacherHealth.AVAILABLE_AUTHENTICATED, version,
                            f"apiProvider={status.get('apiProvider')}",
                            auth_method=auth_method, metered_warnings=warnings,
                            metered_blocking=blocking)

    def _generate(self, system: str, user: str, schema: dict | None) -> str:
        with tempfile.TemporaryDirectory(prefix="docmind-claude-") as workdir:
            argv = [
                self.binary, "-p",
                "--tools", "",              # no Read/Write/Bash: pure text in, text out
                "--output-format", "json",
                "--strict-mcp-config",
                "--system-prompt", system,
            ]
            if self.model:
                argv += ["--model", self.model]
            if schema:
                argv += ["--json-schema", json.dumps(schema)]

            proc = _run(argv, stdin=user, timeout=self.timeout, cwd=workdir)
            if proc.returncode != 0:
                # A quota problem is a *failure*; a success that merely mentions
                # rate limiting is documentation, not an error.
                detail = failure_text(proc)
                if looks_like_usage_limit(detail):
                    raise TeacherUsageLimit(f"claude usage limit: {detail[:400]}")
                raise TeacherCallError(
                    f"claude -p failed (exit {proc.returncode}): {detail[:600]}"
                )
            return _unwrap_claude_result(proc.stdout or "")


def _unwrap_claude_result(stdout: str) -> str:
    """`--output-format json` wraps the answer; hand back the payload itself."""
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout
    if isinstance(envelope, dict):
        for key in ("result", "content", "text", "structured_output"):
            if key in envelope and envelope[key] is not None:
                value = envelope[key]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return stdout
