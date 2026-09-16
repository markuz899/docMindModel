#!/usr/bin/env python
"""Run real test-set prompts through an exported GGUF and record the timings.

    python scripts/validate_gguf.py --model artifacts/export/gguf-0.5b/merged-0.5b-Q4_K_M.gguf

Uses node-llama-cpp, not llama-cli: it is what the DocMind Electron app loads
the model with, so the numbers describe the path that actually ships. Point
--node-modules at any project that has node-llama-cpp installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve_path
from src.dataset.io import read_jsonl
from src.evaluation.metrics import aggregate, evaluate_example
from src.prompt import SYSTEM_PROMPT, build_user_message

DEFAULT_NODE_MODULES = "../docmind"
MIN_NODE_MAJOR = 20


def find_node() -> str | None:
    """A Node new enough for node-llama-cpp's import attributes.

    Node 18 parses `import x from './y.json' with {type: 'json'}` as a syntax
    error, so `node` on PATH is not necessarily usable even when it exists.
    """
    candidates = [shutil.which("node")] if shutil.which("node") else []
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.exists():
        candidates += [str(p / "bin" / "node") for p in sorted(nvm.iterdir(), reverse=True)]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        try:
            version = subprocess.run([candidate, "--version"], capture_output=True,
                                     text=True, timeout=20).stdout.strip()
            if int(version.lstrip("v").split(".")[0]) >= MIN_NODE_MAJOR:
                return candidate
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the .gguf")
    parser.add_argument("--suite", default="data/test/test.jsonl")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--node-modules", default=DEFAULT_NODE_MODULES,
                        help="a project directory containing node-llama-cpp")
    parser.add_argument("--out", default="artifacts/eval/gguf_validation.json")
    parser.add_argument("--node", default=None,
                        help="node binary; defaults to a Node >=20 found on the system")
    args = parser.parse_args()

    node = args.node or find_node()
    if not node:
        raise SystemExit(
            "no Node >= 20 found. node-llama-cpp's dependencies use import "
            "attributes, which Node 18 cannot parse. Install Node 20+ or pass "
            "--node /path/to/node."
        )
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    modules_root = Path(args.node_modules).expanduser()
    if not modules_root.is_absolute():
        modules_root = (resolve_path(".") / modules_root).resolve()
    if not (modules_root / "node_modules" / "node-llama-cpp").exists():
        raise SystemExit(
            f"node-llama-cpp not found under {modules_root}/node_modules.\n"
            "Install it there, or pass --node-modules <project with node-llama-cpp>."
        )

    examples = read_jsonl(args.suite)[: args.examples]
    prompts = [
        {
            "id": e.id,
            "question": e.question,
            "system": SYSTEM_PROMPT,
            "user": build_user_message(e.question, [c.model_dump() for c in e.context]),
        }
        for e in examples
    ]

    with tempfile.TemporaryDirectory() as tmp:
        prompt_file = Path(tmp) / "prompts.json"
        prompt_file.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        argv = [
            node, str(resolve_path("scripts/gguf_bench.mjs")),
            "--model", str(model_path),
            "--prompts", str(prompt_file),
            "--max-tokens", str(args.max_tokens),
            "--context", str(args.context),
            "--module-path", str(modules_root),
        ]
        print("running:", " ".join(argv[:6]), "...")
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"node-llama-cpp failed:\n{proc.stderr[-2000:]}")

    report = json.loads(proc.stdout)

    # Score the GGUF's answers with the same evaluators the PyTorch model uses,
    # so quantization loss shows up as a metric rather than a vibe.
    by_id = {e.id: e for e in examples}
    rows = [evaluate_example(by_id[r["id"]], r["answer"]) for r in report["results"]
            if r["id"] in by_id]
    report["quality"] = aggregate(rows)

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ngpu={report['gpu']} load={report['load_ms']}ms "
          f"ttft_median={report['median_ttft_ms']}ms "
          f"chunks/s_median={report['median_chunks_per_second']} "
          f"rss={report['process_rss_mib']}MiB")
    print(f"quality on {len(rows)} prompt(s): "
          f"hallucination={report['quality'].get('hallucinated_rate')} "
          f"useful_answer_rate={report['quality'].get('useful_answer_rate')}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
