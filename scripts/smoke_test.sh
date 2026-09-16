#!/usr/bin/env bash
# End-to-end smoke test: every stage of the pipeline, in order, in ~1 minute.
#
#   ./scripts/smoke_test.sh
#
# It uses a randomly initialised ~0.3M-parameter model built locally, so it
# downloads nothing and trains nothing expensive. The answers are gibberish on
# purpose: what this proves is that the pipeline runs, not that it is good.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "1/8  unit tests"
"$PY" -m pytest tests/ -q

step "2/8  build dataset from the Markdown corpus + seeds"
"$PY" scripts/build_dataset.py

step "3/8  build the targeted benchmarks"
"$PY" scripts/build_benchmarks.py

step "4/8  validate every dataset file against the schema + quality gate"
"$PY" scripts/validate_dataset.py --all

step "5/8  teacher generation pipeline (mock provider, no API key needed)"
"$PY" scripts/generate_dataset.py --docs data/raw/projects --provider mock \
      --limit 3 --out data/generated/smoke-candidates.jsonl

step "6/8  build the tiny offline model"
"$PY" scripts/make_tiny_model.py

step "7/8  training smoke test (4 steps)"
"$PY" -m src.training.train --smoke

step "8/8  inference + evaluation smoke test"
"$PY" -m src.inference.chat --model-config configs/model.smoke.yaml \
      --adapter artifacts/smoke-adapter --context data/raw/projects/acme-identity \
      --max-sources 3 --max-new-tokens 16 \
      --question "Come funziona GET /me?"
"$PY" -m src.evaluation.evaluate --model-config configs/model.smoke.yaml \
      --adapter artifacts/smoke-adapter --suites test,hallucination \
      --limit 4 --batch-size 4 --max-new-tokens 16 \
      --out artifacts/eval/smoke_results.json
"$PY" scripts/evaluate_models.py --model-config configs/model.smoke.yaml \
      --adapter artifacts/smoke-adapter --suites test --limit 4 \
      --max-new-tokens 16 --out artifacts/eval/smoke_comparison.json

printf '\n\033[1;32mAll stages completed.\033[0m See artifacts/ for outputs.\n'
