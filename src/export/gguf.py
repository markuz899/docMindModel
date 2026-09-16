#!/usr/bin/env python
"""Convert a merged model to GGUF and quantize it for llama.cpp.

    python -m src.export.gguf --model artifacts/export/merged --test

Needs a llama.cpp checkout/install. It is found via, in order:
  --llama-cpp, $LLAMA_CPP_DIR, llama-quantize on PATH, ./vendor/llama.cpp,
  ~/llama.cpp. If nothing is found the script prints exactly what to install
  and exits 2 -- it never pretends the export happened.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_model_config, resolve_path

SETUP_HINT = """
llama.cpp was not found. Either:

  git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
  cmake -B vendor/llama.cpp/build -S vendor/llama.cpp
  cmake --build vendor/llama.cpp/build --config Release -j
  pip install -r vendor/llama.cpp/requirements.txt

or point the script at an existing checkout:

  python -m src.export.gguf --llama-cpp /path/to/llama.cpp
  LLAMA_CPP_DIR=/path/to/llama.cpp python -m src.export.gguf
"""


def find_llama_cpp(explicit: str | None) -> dict[str, Path | None]:
    candidates = [
        Path(explicit) if explicit else None,
        Path(os.environ["LLAMA_CPP_DIR"]) if os.environ.get("LLAMA_CPP_DIR") else None,
        resolve_path("vendor/llama.cpp"),
        Path.home() / "llama.cpp",
    ]
    convert = quantize = cli = None
    for root in [c for c in candidates if c and c.exists()]:
        convert = convert or next(
            (p for p in [root / "convert_hf_to_gguf.py", root / "convert-hf-to-gguf.py"]
             if p.exists()), None
        )
        for rel in ("build/bin/llama-quantize", "llama-quantize", "build/bin/quantize"):
            if (root / rel).exists():
                quantize = quantize or root / rel
        for rel in ("build/bin/llama-cli", "llama-cli", "main"):
            if (root / rel).exists():
                cli = cli or root / rel
    # installed system-wide (brew install llama.cpp, apt, ...)
    quantize = quantize or (Path(shutil.which("llama-quantize")) if shutil.which("llama-quantize") else None)
    cli = cli or (Path(shutil.which("llama-cli")) if shutil.which("llama-cli") else None)
    return {"convert": convert, "quantize": quantize, "cli": cli}


def warn_if_emulated(binary: Path) -> None:
    """On Apple Silicon, an x86_64 llama.cpp runs under Rosetta: no Metal, ~10-50x
    slower, and GGUF smoke tests will time out. Worth saying out loud."""
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return
    try:
        archs = subprocess.check_output(["lipo", "-archs", str(binary)], text=True).split()
    except Exception:
        return
    if "arm64" not in archs:
        print(
            f"WARNING: {binary} is {' '.join(archs)} only. On Apple Silicon it runs "
            "under Rosetta, without Metal. Build llama.cpp from source, or install "
            "an arm64 Homebrew (/opt/homebrew), before trusting any timing."
        )


def _run(cmd: list[str]) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def convert_to_gguf(model_dir: Path, out_dir: Path, convert_script: Path, outtype: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{model_dir.name}-{outtype}.gguf"
    _run([sys.executable, convert_script, model_dir, "--outfile", target, "--outtype", outtype])
    return target


def quantize(source: Path, quantize_bin: Path, quant_type: str, stem: str) -> Path:
    target = source.with_name(f"{stem}-{quant_type}.gguf")
    _run([quantize_bin, source, target, quant_type])
    return target


def run_test_prompts(
    cli_bin: Path, gguf: Path, n: int, max_tokens: int, timeout: int = 300
) -> list[dict]:
    """Run a few real test-set prompts through llama.cpp and show the output."""
    from src.dataset.io import read_jsonl
    from src.prompt import SYSTEM_PROMPT, build_user_message

    examples = read_jsonl("data/test/test.jsonl")[:n]
    results = []
    for ex in examples:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{build_user_message(ex.question, [c.model_dump() for c in ex.context])}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        try:
            proc = subprocess.run(
                [str(cli_bin), "-m", str(gguf), "-p", prompt, "-n", str(max_tokens),
                 "--temp", "0", "-no-cnv"],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"\n--- {ex.id}: TIMED OUT after {timeout}s")
            results.append({"id": ex.id, "question": ex.question, "answer": None,
                            "returncode": None, "timeout_s": timeout})
            continue
        answer = proc.stdout.replace(prompt, "").strip()
        print(f"\n--- {ex.id}: {ex.question}")
        print(answer[:800] or "(no output)")
        results.append({"id": ex.id, "question": ex.question, "answer": answer,
                        "returncode": proc.returncode})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model", default=None, help="merged model dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--llama-cpp", default=None)
    parser.add_argument("--quantizations", default=None, help="comma separated, e.g. Q4_K_M,Q8_0")
    parser.add_argument("--outtype", default=None)
    parser.add_argument("--test", action="store_true", help="run prompts through the GGUF")
    parser.add_argument("--test-examples", type=int, default=3)
    parser.add_argument("--test-max-tokens", type=int, default=256)
    parser.add_argument("--test-timeout", type=int, default=300,
                        help="seconds per test prompt before giving up")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config(args.model_config)
    model_dir = resolve_path(args.model or cfg.export.merged_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"{model_dir} is not a model directory -- run src.export.merge first")
    out_dir = resolve_path(args.out or cfg.export.gguf_dir)
    outtype = args.outtype or cfg.export.outtype
    quants = (
        [q.strip() for q in args.quantizations.split(",") if q.strip()]
        if args.quantizations
        else list(cfg.export.quantizations)
    )

    tools = find_llama_cpp(args.llama_cpp)
    if not tools["convert"] or not tools["quantize"]:
        print(f"convert script: {tools['convert']}\nquantize binary: {tools['quantize']}")
        print(SETUP_HINT)
        return 2

    print(f"[gguf] converting {model_dir} -> {outtype}")
    base_gguf = convert_to_gguf(model_dir, out_dir, tools["convert"], outtype)

    produced = {outtype: base_gguf}
    for quant_type in quants:
        print(f"[gguf] quantizing {quant_type}")
        produced[quant_type] = quantize(base_gguf, tools["quantize"], quant_type, model_dir.name)

    if not args.keep_intermediate and len(produced) > 1 and base_gguf.exists():
        base_gguf.unlink()
        produced.pop(outtype)

    primary = produced.get(cfg.export.primary_quantization) or next(iter(produced.values()))
    tests = []
    if args.test:
        if not tools["cli"]:
            print("[gguf] llama-cli not found; skipping the GGUF smoke test")
        else:
            warn_if_emulated(tools["cli"])
            print(f"[gguf] testing {primary.name}")
            tests = run_test_prompts(tools["cli"], primary, args.test_examples,
                                     args.test_max_tokens, args.test_timeout)

    source_metadata = model_dir / "model_metadata.json"
    metadata = json.loads(source_metadata.read_text(encoding="utf-8")) if source_metadata.exists() else {}
    metadata.update({
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gguf_dir": str(out_dir),
        "primary_quantization": cfg.export.primary_quantization,
        "quantizations": {
            name: {"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 1)}
            for name, path in produced.items()
        },
        "gguf_smoke_test": tests or None,
        "llama_cpp": {k: str(v) for k, v in tools.items() if v},
    })
    (out_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[gguf] artefacts in {out_dir}:")
    for name, path in produced.items():
        print(f"  {name:>8}  {path.stat().st_size / 1e6:8.1f} MB  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
