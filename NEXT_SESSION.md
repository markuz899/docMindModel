# Resume here

State at the end of 2026-09-16. Six commits, clean tree, 119 tests passing,
nothing running.

```bash
cd docmind-model
.venv/bin/python -m pytest tests/ -q          # expect 119 passed
git log --oneline | head -6
```

---

## 1. The one open bug: v2 training diverges

**Do not start a long v2 training run before fixing this.**

Measured, not suspected:

| | eval_loss on `data/real/validation` |
|---|---|
| base Qwen2.5-0.5B, no adapter | **5.233** |
| after 1 epoch of v2 training | **6.200** |

Training makes the model *worse*. v1 on demo data went 1.94 → 0.38 smoothly,
so this is specific to the real-data run.

Already ruled out by measurement, do not re-check:

* **loss masking is correct** — decoding the supervised tokens of a real
  example returns exactly the answer, `<|im_end|>` included;
* **LoRA params are already fp32** — PEFT upcasts them, so this is not
  unscaled pure-fp16 adapter training.

The live hypothesis, and why:

* `grad_norm` runs 10-12 against ~2 in the v1 run, with `max_grad_norm: 1.0`,
  so nearly every update is being clipped by an order of magnitude;
* real answers are short and contexts long, so **only 5-7% of tokens are
  supervised** (69 of 1307 in one example) against 18-19% in the demo data.
  With `gradient_accumulation_steps: 4` that is ~280 supervised tokens per
  optimizer step — a very noisy gradient at `learning_rate: 2e-4`.

Next step, in this order:

```bash
# probe 12 steps at a lower LR and a larger batch, watch the loss trend
sed -e 's|^learning_rate: .*|learning_rate: 5.0e-5|' \
    -e 's|^gradient_accumulation_steps: .*|gradient_accumulation_steps: 8|' \
    -e 's|^logging_steps: .*|logging_steps: 2|' \
    configs/training.v2-real.yaml > /tmp/probe-lr.yaml

.venv/bin/python -u -m src.training.train \
    --model-config configs/model.local-0.5b.yaml \
    --training-config /tmp/probe-lr.yaml \
    --dataset-config configs/dataset.real.yaml \
    --max-steps 12 --output-dir artifacts/probe-lr
```

Success criterion: loss trending below 5.23 (the base model's score). The probe
was started once and killed for being slow — at 1536 tokens with accumulation 8
a step takes ~200 s, so 12 steps is ~40 minutes. Run it in the background.

If a lower LR does not fix it, the next suspects are MPS fp16 numerics at 1536
tokens (try `dtype: float32` with `max_seq_length: 1024` to trade coverage for
stability) and the tiny supervised-token density (more data fixes that by
itself).

---

## 2. Finish the dataset

213 of a planned ~1600 examples. The Codex quota ran out at bundle 51 of 322;
it resets 2026-09-17 03:35.

Nothing already generated is re-requested — 53 cached responses are served for
free, and the run state records the 51 finished bundles.

```bash
# Codex, once the quota is back
.venv/bin/python -m src.generation.generate --docs artifacts/corpus/real \
    --teacher codex --concurrency 2 --resume \
    --out artifacts/corpus/real-dataset.jsonl

# or Claude, which is now usable on the Pro subscription (see commit ae2daaf)
.venv/bin/python -m src.generation.generate --docs artifacts/corpus/real \
    --teacher claude --concurrency 2 --resume \
    --out artifacts/corpus/real-dataset.jsonl
```

Check first, it costs nothing:

```bash
.venv/bin/python -m src.generation.generate --health-check
# expect: "Usable on a subscription: codex, claude"
```

Expect ~40-50 s per call, so 271 remaining bundles is 1.5-2 h at concurrency 2.
A usage limit stops the run cleanly; re-run the same command with `--resume`.

Then rebuild the splits:

```bash
.venv/bin/python scripts/build_real_dataset.py \
    --from artifacts/corpus/real-dataset.jsonl --out-dir data/real --version 2.0.0
```

---

## 3. Train v2, compare, export

Only after 1 and 2.

```bash
.venv/bin/python -u -m src.training.train \
    --model-config configs/model.local-0.5b.yaml \
    --training-config configs/training.v2-real.yaml \
    --dataset-config configs/dataset.real.yaml

# Base vs v1 vs v2 on the same real test set
.venv/bin/python -m src.evaluation.evaluate --model-config configs/model.local-0.5b.yaml \
    --suites data/real/test/test.jsonl --out artifacts/eval/real_base.json
.venv/bin/python -m src.evaluation.evaluate --model-config configs/model.local-0.5b.yaml \
    --adapter artifacts/adapter-0.5b --suites data/real/test/test.jsonl \
    --out artifacts/eval/real_v1.json
.venv/bin/python -m src.evaluation.evaluate --model-config configs/model.local-0.5b.yaml \
    --adapter artifacts/v2-real/adapter --suites data/real/test/test.jsonl \
    --out artifacts/eval/real_v2.json

# export only if v2 beats v1 on Useful Answer Rate
.venv/bin/python -m src.export.merge --model-config configs/model.local-0.5b.yaml \
    --adapter artifacts/v2-real/adapter --out artifacts/v2-real/merged
.venv/bin/python -m src.export.gguf --model-config configs/model.local-0.5b.yaml \
    --model artifacts/v2-real/merged --out artifacts/v2-real/gguf
.venv/bin/python scripts/validate_gguf.py --model artifacts/v2-real/gguf/merged-Q4_K_M.gguf \
    --suite data/real/test/test.jsonl --examples 20
```

Then write `REAL_RESULTS.md` (not yet started) with the Base/v1/v2 table.
`RESULTS.md` and every v1 artefact stay untouched.

---

## 4. Why v1 must not ship

Measured on the user's own documentation, through node-llama-cpp — the path the
Electron app uses (`artifacts/eval/gguf_v1_on_real_docs.json`):

| metric | value |
|---|---|
| hallucination | 0.05 |
| **useful answer rate** | **0.00** |
| over-refusal | 0.82 |
| citation F1 | 0.00 |

Confusion matrix: TP=3, **FN=14**, FP=0, TN=3. Of 17 answerable questions it
answered none usefully and refused 14, reproducing the refusal template
verbatim — guillemets included — even when the cited heading was the answer.

The low hallucination rate is the trap Useful Answer Rate was added to expose:
v1 scores 0.05 because it refuses almost everything.

v1 is fine for exercising the Electron `LocalGGUFProvider` (load, Metal,
streaming, prompt format). It is not fine as a product model.

Performance, for reference: `gpu: metal`, load 805-887 ms, TTFT 212-608 ms,
59-76 chunks/s, 287-515 MiB RSS, 398 MB on disk.

---

## Open decisions for the user

1. **Privacy.** `data/real/` and `artifacts/corpus/` contain documentation from
   eight of your projects (odysseus, homehub, botforge, nltAi, logAi,
   nightly-doc-worker, iad-emission-manager-fastify, afu-builder). Only
   `data/real/` is inside a git-tracked path; nothing has been committed yet.
   Decide whether the v2 dataset should be versioned or stay local.
2. **Corpus size.** 322 sections yields ~1300-1600 examples, below the
   2000-4000 target. More projects, or more context combinations per section
   (`generation.max_contexts`), would close the gap.
3. **Manual benchmark.** The scaffold exists and is wired to hold its ids out
   of training, but no items have been reviewed:
   `python scripts/manual_benchmark.py build --from artifacts/corpus/real-dataset.jsonl --size 150`

## Where things are

| | |
|---|---|
| real corpus (8 projects, 322 sections) | `artifacts/corpus/real/` |
| generated candidates + run state | `artifacts/corpus/real-dataset.jsonl` |
| v2 splits (train 161 / val 21 / test 31) | `data/real/` |
| teacher cache (53 requests) | `data/cache/teacher/` |
| v1 adapter and GGUF | `artifacts/adapter-0.5b/`, `artifacts/export/gguf-0.5b/` |
| GGUF copied for the app | `../models/docmind-0.5b-v1-demo-Q4_K_M.gguf` |
| teacher CLI facts and costs | `docs/teacher-cli.md` |
