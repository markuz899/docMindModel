#!/usr/bin/env python
"""Check whether this machine can run the pipeline, and what it can run.

    python scripts/doctor.py

Run it first on any new machine. It reports what is present, what is missing,
and which training profile fits the memory it finds -- so the answer to "can I
train the 1.5B here?" is measured rather than guessed.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, WARN, BAD = "ok", "warn", "missing"
_MARK = {OK: "  ok  ", WARN: " warn ", BAD: "MISSING"}

REQUIRED = ["torch", "transformers", "datasets", "peft", "accelerate", "pydantic", "pyyaml"]
OPTIONAL = {
    "bitsandbytes": "QLoRA 4-bit training (Linux + CUDA only)",
    "sentencepiece": "GGUF conversion (llama.cpp's convert script)",
    "gguf": "GGUF conversion",
}


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{_MARK[status]}] {label:<26} {detail}")


def total_ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / 1e9, 1)
        if platform.system() == "Linux":
            for row in Path("/proc/meminfo").read_text().splitlines():
                if row.startswith("MemTotal:"):
                    return round(int(row.split()[1]) / 1e6, 1)
    except Exception:
        return None
    return None


def recommend(ram: float | None, device: str, vram_gb: float | None) -> None:
    print("\n--- what this machine can train ---")
    budget = vram_gb if device == "cuda" and vram_gb else ram
    if budget is None:
        print("  could not determine memory; start with the 0.5B profile")
        return

    # The dominant tensor is the loss input, seq_len x vocab_size (151936 for
    # Qwen2.5), so sequence length drives memory harder than parameter count.
    if budget >= 24:
        print(f"  {budget} GB -> 1.5B at max_seq_length 2048+, or 0.5B unconstrained")
        print("     configs/model.yaml + configs/training.yaml")
    elif budget >= 14:
        print(f"  {budget} GB -> 0.5B at max_seq_length 2048 in float32 (96% of real examples)")
        print("     configs/model.local-0.5b-16gb.yaml + configs/training.v2-16gb.yaml")
        print("     1.5B is feasible at max_seq_length 1024 with gradient checkpointing")
    elif budget >= 7:
        print(f"  {budget} GB -> 0.5B at max_seq_length 1536 in float16 only")
        print("     configs/model.local-0.5b.yaml + configs/training.v2-real.yaml")
        print("     close other applications first; swap thrashing is the usual failure")
    else:
        print(f"  {budget} GB -> smoke tests only (configs/*.smoke.yaml)")


def main() -> int:
    print(f"docmind-model doctor -- {platform.platform()}")
    print(f"python {platform.python_version()} ({sys.executable})\n")

    if sys.version_info < (3, 10):
        line(BAD, "python >= 3.10", f"found {platform.python_version()}")
    else:
        line(OK, "python", platform.python_version())

    ram = total_ram_gb()
    line(OK if (ram or 0) >= 14 else WARN, "system RAM", f"{ram} GB" if ram else "unknown")

    free_gb = round(shutil.disk_usage(".").free / 1e9, 1)
    line(OK if free_gb >= 20 else WARN, "free disk", f"{free_gb} GB (weights + GGUF need ~10 GB)")

    for name in REQUIRED:
        try:
            line(OK, name, md.version(name))
        except md.PackageNotFoundError:
            line(BAD, name, "pip install -r requirements.txt")
    for name, why in OPTIONAL.items():
        try:
            line(OK, name, md.version(name))
        except md.PackageNotFoundError:
            line(WARN, name, f"optional: {why}")

    device, vram = "cpu", None
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            line(OK, "accelerator", f"cuda: {torch.cuda.get_device_name(0)}, {vram} GB VRAM")
            line(OK if torch.cuda.is_bf16_supported() else WARN, "bf16",
                 "supported" if torch.cuda.is_bf16_supported() else "not supported, fp16 will be used")
        elif torch.backends.mps.is_available():
            device = "mps"
            line(OK, "accelerator", "mps (Apple Silicon)")
            line(WARN, "mps notes",
                 "fp32 training is forced when dtype: auto; keep empty_cache_steps: 1")
        else:
            line(WARN, "accelerator", "cpu only -- training will be slow")
    except ImportError:
        line(BAD, "torch", "pip install -r requirements.txt")

    print()
    for binary, probe in (("codex", ["codex", "login", "status"]),
                          ("claude", ["claude", "auth", "status", "--json"])):
        if not shutil.which(binary):
            line(WARN, f"teacher: {binary}", "not installed (needed only to generate data)")
            continue
        try:
            proc = subprocess.run(probe, capture_output=True, text=True, timeout=45)
            authed = proc.returncode == 0 and "not logged in" not in proc.stdout.lower()
            line(OK if authed else WARN, f"teacher: {binary}",
                 "authenticated" if authed else f"run `{binary} login`")
        except Exception as exc:
            line(WARN, f"teacher: {binary}", str(exc)[:60])

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(var):
            line(WARN, f"env: {var}", "set; stripped from teacher subprocesses by default")

    node = shutil.which("node")
    if node:
        version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        major = int(version.lstrip("v").split(".")[0] or 0)
        line(OK if major >= 20 else WARN, "node", f"{version} (>= 20 needed for GGUF validation)")
    else:
        line(WARN, "node", "not installed (needed only to validate a GGUF)")

    print()
    for label, path in (("v2 corpus", "artifacts/corpus/real"),
                        ("v2 candidates", "artifacts/corpus/real-dataset.jsonl"),
                        ("teacher cache", "data/cache/teacher"),
                        ("v1 adapter", "artifacts/adapter-0.5b"),
                        ("v1 GGUF", "artifacts/export/gguf-0.5b")):
        target = Path(path)
        if not target.exists():
            line(WARN, label, f"absent ({path})")
        elif target.is_dir():
            count = sum(1 for _ in target.rglob("*") if _.is_file())
            line(OK, label, f"{count} file(s)")
        else:
            line(OK, label, f"{sum(1 for _ in target.open()) } line(s)")

    recommend(ram, device, vram)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
