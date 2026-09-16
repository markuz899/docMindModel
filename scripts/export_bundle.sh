#!/usr/bin/env bash
# Package what is not in git but is expensive or impossible to recreate.
#
#   ./scripts/export_bundle.sh [output.tar.gz]          # essentials, a few MB
#   ./scripts/export_bundle.sh --full [output.tar.gz]   # + weights, ~1.8 GB
#
# The essentials are small and irreplaceable. The teacher cache in particular is
# subscription quota already spent: losing it costs hours of regeneration, and
# the dataset cannot be reproduced byte-for-byte without it.
#
# --full adds the v1 adapter, the v1 GGUF and the downloaded base model. All
# three can be recreated -- the base model with one huggingface-cli command, the
# GGUF by re-running merge + export -- so move them only if bandwidth is cheaper
# than the rebuild.
set -euo pipefail
cd "$(dirname "$0")/.."

FULL=0
if [ "${1:-}" = "--full" ]; then FULL=1; shift; fi
OUT="${1:-docmind-model-bundle-$(date +%Y%m%d).tar.gz}"

ESSENTIAL=(data/cache/teacher data/real artifacts/corpus)
HEAVY=(artifacts/adapter-0.5b artifacts/export/gguf-0.5b artifacts/base)

PATHS=()
for candidate in "${ESSENTIAL[@]}"; do [ -e "$candidate" ] && PATHS+=("$candidate"); done
if [ "$FULL" = "1" ]; then
  for candidate in "${HEAVY[@]}"; do [ -e "$candidate" ] && PATHS+=("$candidate"); done
fi

if [ ${#PATHS[@]} -eq 0 ]; then echo "nothing to bundle" >&2; exit 1; fi

echo "bundling:"
for p in "${PATHS[@]}"; do printf '  %-34s %s\n' "$p" "$(du -sh "$p" | cut -f1)"; done
tar -czf "$OUT" "${PATHS[@]}"
echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"

if [ "$FULL" = "0" ]; then
  cat <<'NOTE'

Not included (recreate on the new machine):
  artifacts/base          huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
                              --local-dir artifacts/base/Qwen2.5-0.5B-Instruct
  artifacts/adapter-0.5b  only needed to reproduce the v1 comparison; use --full
  artifacts/export        re-export with src.export.merge + src.export.gguf
NOTE
fi

cat <<NOTE

On the new machine:
  git clone <this repo> docmind-model && cd docmind-model
  tar -xzf $(basename "$OUT") -C .
  python -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python scripts/doctor.py
  .venv/bin/python -m pytest tests/ -q      # expect 119 passed
NOTE
