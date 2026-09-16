# Teacher CLIs

Dataset generation runs through a coding CLI you are already logged into, not
through a metered API. This file records what was verified, on which versions,
so the next person does not have to rediscover it.

Verified 2026-09-16 on macOS 14 / Apple M1:

| CLI | Version | Non-interactive mode | Structured output | Free auth probe |
|---|---|---|---|---|
| `codex` | codex-cli 0.154.0 | `codex exec` | `--output-schema FILE` | `codex login status` |
| `claude` | 2.1.273 | `claude -p` | `--json-schema JSON` | `claude auth status --json` |

**Do not assume these flags.** Both CLIs move fast. Run `codex exec --help` and
`claude --help` before changing `src/generation/cli_teachers.py`.

## How each one is invoked

```bash
codex exec --sandbox read-only --skip-git-repo-check --ephemeral \
           --color never --cd <empty temp dir> \
           --output-last-message <file> --output-schema <schema.json> -
# prompt on stdin; the answer is read from --output-last-message, because
# codex echoes the entire prompt to stderr before it answers.

claude -p --tools "" --output-format json --strict-mcp-config \
          --system-prompt <text> --json-schema <json>
# prompt on stdin; `--tools ""` disables every tool, so it is text in, text out.
```

Both run with the working directory set to a throwaway temp directory. The
teacher sees the documentation blocks of the current batch and nothing else --
not this repository, not your code.

## Billing safety

Both CLIs can authenticate either against a subscription or against a metered
API key, and the second is easy to trigger by accident: a shell with
`ANTHROPIC_API_KEY` exported will bill Claude Code per token even when you are
logged in with `claude.ai`.

The pipeline therefore:

* treats `codex` logged in with an API key, or `claude` reporting a non-null
  `apiKeySource`, as a **blocking** billing risk and refuses to run until
  `--allow-metered-env` is passed;
* mentions `OPENAI_API_KEY` as advisory only, because `codex login status`
  reports the ChatGPT login and that takes precedence;
* never selects `openai` or `anthropic` under `--teacher auto`;
* never falls back to a metered API when a CLI fails. Running out of
  subscription quota stops the run; it does not start charging you.

It never reads, copies or writes credentials. `claude --bare` is deliberately
unused: it forces API-key authentication by design.

```bash
python -m src.generation.generate --health-check
```

## Reading failures correctly

`codex exec` writes its banner *and the whole prompt* to stderr before
answering. Scanning that output for "rate limit" therefore reads a gateway's
own documentation as a quota error -- which is exactly what happened on the
first real run. Only the tail of each stream is inspected, and only when the
process exited non-zero.

Usage limits are recognised, checkpointed and reported; they are not retried.
Re-run the same command with `--resume` when quota returns.

## Cost of a run

Measured on the corpora in this repository, `codex exec` with a batch of 5:

| | |
|---|---|
| time per call | 40-50 s |
| examples per call | 5 requested, ~4.1 accepted after the quality gate |
| 322 sections | 322 calls, ~1.5-2 h at concurrency 2 |

Every request is cached on provider, CLI version, model, prompt version,
documentation content and generation options, so a re-run costs nothing. That
is not an optimisation: it is what makes it safe to fix the quality gate and
re-score 300 calls' worth of output without spending quota again.
