"""Teacher CLI providers. The real `codex`/`claude` binaries are never invoked:
every test fakes the subprocess, so the suite consumes no subscription quota."""

import json
import subprocess

import pytest

from src.generation import cli_teachers as cli
from src.generation.cache import RunState, TeacherCache
from src.generation.cli_teachers import (
    ClaudeCodeCliTeacher,
    CodexCliTeacher,
    TeacherCallError,
    TeacherHealth,
    TeacherUsageLimit,
    failure_text,
    looks_like_usage_limit,
)


def _proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


@pytest.fixture
def fake_cli(monkeypatch):
    """Replace subprocess execution with a scripted response per command."""
    calls = []

    def factory(responses):
        def fake_run(argv, stdin=None, timeout=300, cwd=None):
            calls.append({"argv": argv, "stdin": stdin, "cwd": cwd})
            for match, response in responses.items():
                if match in " ".join(argv):
                    return response(argv) if callable(response) else response
            return _proc(stdout="{}")
        monkeypatch.setattr(cli, "_run", fake_run)
        monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/local/bin/{name}")
        return calls

    return factory


# --- detection and health -------------------------------------------------

def test_missing_binary_is_reported_not_installed(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    report = CodexCliTeacher().health_check()
    assert report.status is TeacherHealth.NOT_INSTALLED
    assert not report.usable


def test_codex_logged_in_with_chatgpt_is_authenticated(fake_cli):
    fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(stdout="Logged in using ChatGPT"),
    })
    report = CodexCliTeacher().health_check()
    assert report.usable and report.auth_method == "ChatGPT"
    assert not report.metered_blocking


def test_codex_logged_out_is_not_authenticated(fake_cli):
    fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(returncode=1, stdout="Not logged in"),
    })
    assert CodexCliTeacher().health_check().status is TeacherHealth.AVAILABLE_NOT_AUTHENTICATED


def test_codex_api_key_login_is_a_billing_risk(fake_cli):
    fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(stdout="Logged in using an API key"),
    })
    report = CodexCliTeacher().health_check()
    assert report.usable and report.metered_blocking


def test_claude_subscription_login_is_authenticated(fake_cli, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_cli({
        "--version": _proc(stdout="2.1.273 (Claude Code)"),
        "auth status": _proc(stdout=json.dumps(
            {"loggedIn": True, "authMethod": "claude.ai",
             "apiProvider": "firstParty", "apiKeySource": None})),
    })
    report = ClaudeCodeCliTeacher().health_check()
    assert report.usable and not report.metered_blocking


def test_claude_api_key_source_blocks_generation(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_cli({
        "--version": _proc(stdout="2.1.273 (Claude Code)"),
        "auth status": _proc(stdout=json.dumps(
            {"loggedIn": True, "authMethod": "claude.ai",
             "apiProvider": "firstParty", "apiKeySource": "ANTHROPIC_API_KEY"})),
    })
    teacher = ClaudeCodeCliTeacher()
    assert teacher.health_check().metered_blocking
    with pytest.raises(TeacherCallError, match="billed per token"):
        teacher.complete("sys", "user")


def test_allow_metered_env_overrides_the_block(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    fake_cli({
        "--version": _proc(stdout="2.1.273"),
        "auth status": _proc(stdout=json.dumps(
            {"loggedIn": True, "authMethod": "claude.ai", "apiKeySource": "ANTHROPIC_API_KEY"})),
        "-p": _proc(stdout=json.dumps({"result": '{"examples": []}'})),
    })
    out = ClaudeCodeCliTeacher(allow_metered_env=True).complete("sys", "user")
    assert json.loads(out) == {"examples": []}


# --- invocation shape -----------------------------------------------------

def test_codex_runs_read_only_and_never_sees_this_repository(fake_cli, tmp_path):
    calls = fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(stdout="Logged in using ChatGPT"),
        "exec": _proc(stdout='{"examples": []}'),
    })
    CodexCliTeacher().complete("system text", "user text", {"type": "object"})
    exec_call = next(c for c in calls if "exec" in c["argv"])
    argv = exec_call["argv"]
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in argv and "--skip-git-repo-check" in argv
    assert "--output-schema" in argv
    # the working directory is a throwaway temp dir, not the repo
    assert exec_call["cwd"] and "docmind-codex-" in exec_call["cwd"]
    assert "system text" in exec_call["stdin"] and "user text" in exec_call["stdin"]


def test_claude_runs_with_all_tools_disabled(fake_cli):
    calls = fake_cli({
        "--version": _proc(stdout="2.1.273"),
        "auth status": _proc(stdout=json.dumps({"loggedIn": True, "authMethod": "claude.ai"})),
        "-p": _proc(stdout=json.dumps({"result": '{"examples": []}'})),
    })
    ClaudeCodeCliTeacher().complete("sys", "user")
    call = next(c for c in calls if "-p" in c["argv"])
    assert call["argv"][call["argv"].index("--tools") + 1] == ""
    assert "--strict-mcp-config" in call["argv"]
    # --bare would force ANTHROPIC_API_KEY auth, i.e. per-token billing
    assert "--bare" not in call["argv"]


# --- failure handling -----------------------------------------------------

def test_usage_limit_is_distinguished_from_a_plain_failure(fake_cli):
    fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(stdout="Logged in using ChatGPT"),
        "exec": _proc(returncode=1, stderr="Error: usage limit reached, resets at 14:00"),
    })
    with pytest.raises(TeacherUsageLimit):
        CodexCliTeacher().complete("sys", "user")


def test_ordinary_failure_is_not_mistaken_for_a_usage_limit(fake_cli):
    fake_cli({
        "--version": _proc(stdout="codex-cli 0.154.0"),
        "login status": _proc(stdout="Logged in using ChatGPT"),
        "exec": _proc(returncode=1, stderr='{"code": "invalid_json_schema"}'),
    })
    with pytest.raises(TeacherCallError):
        CodexCliTeacher().complete("sys", "user")


def test_documentation_about_rate_limiting_is_not_a_usage_limit():
    """Regression: `codex exec` echoes the prompt, so a gateway's own docs about
    rate limiting used to be read as a quota error and abort the run."""
    echoed = "The gateway applies rate limits per caller. RateLimiter is a token bucket."
    assert not looks_like_usage_limit(echoed)
    assert looks_like_usage_limit("You have hit your usage limit; resets at 09:00")


def test_failure_text_excludes_the_echoed_prompt():
    huge_prompt = "documentation about rate limits\n" * 500
    proc = _proc(returncode=1, stderr=huge_prompt + "\nERROR: quota exceeded")
    detail = failure_text(proc)
    assert "quota exceeded" in detail
    assert len(detail) < len(huge_prompt)


def test_a_cli_failure_never_falls_back_to_a_metered_api(fake_cli):
    from src.generation.providers import METERED_PROVIDERS, AUTO_ORDER

    assert not METERED_PROVIDERS & set(AUTO_ORDER), "auto must never reach a paid API"


# --- cache and resume -----------------------------------------------------

def test_cache_key_changes_with_anything_that_changes_the_answer():
    base = dict(provider="codex", provider_version="1.0", model=None,
                system="s", user="u", schema=None)
    key = TeacherCache.request_hash(**base)
    assert TeacherCache.request_hash(**{**base, "provider_version": "1.1"}) != key
    assert TeacherCache.request_hash(**{**base, "user": "u2"}) != key
    assert TeacherCache.request_hash(**{**base, "model": "gpt"}) != key
    assert TeacherCache.request_hash(**base, prompt_version="99") != key
    assert TeacherCache.request_hash(**base) == key  # deterministic


def test_cache_round_trip_and_stats(tmp_path):
    cache = TeacherCache(root=tmp_path / "c")
    key = TeacherCache.request_hash("codex", "1.0", None, "s", "u", None)
    assert cache.get(key) is None
    cache.put(key, "payload", {"bundle": "b1"})
    assert cache.get(key) == "payload"
    assert cache.stats.hits == 1 and cache.stats.misses == 1 and cache.entries() == 1


def test_disabled_cache_never_returns_anything(tmp_path):
    cache = TeacherCache(root=tmp_path / "c", enabled=False)
    key = TeacherCache.request_hash("codex", "1.0", None, "s", "u", None)
    cache.put(key, "payload")
    assert cache.get(key) is None


def test_run_state_survives_a_restart(tmp_path):
    state = RunState.load(tmp_path / "s.json", resume=False)
    state.mark("bundle-1")
    state.bump("teacher_calls", 4)
    state.save()

    resumed = RunState.load(tmp_path / "s.json", resume=True)
    assert resumed.done("bundle-1") and resumed.counters["teacher_calls"] == 4
    assert not resumed.done("bundle-2")

    fresh = RunState.load(tmp_path / "s.json", resume=False)
    assert not fresh.done("bundle-1"), "without --resume the run starts over"


def test_usage_limit_message_shows_the_error_not_the_echoed_prompt():
    """Observed: the run stopped correctly but reported someone's documentation.

    codex echoes the whole prompt, so slicing the output by offset shows the
    tail of the prompt. The message must quote the line that matched.
    """
    from src.generation.cli_teachers import usage_limit_excerpt

    noise = "\n".join(f"[doc-{i}.md - Section] documentation body" for i in range(200))
    real = "ERROR: You've hit your usage limit. Try again at Sep 17th, 3:35 AM."
    excerpt = usage_limit_excerpt(f"{noise}\n{real}\n{real}")
    assert "usage limit" in excerpt.lower()
    assert "documentation body" not in excerpt
    assert excerpt.count("hit your usage limit") == 1, "duplicate stream output collapsed"
