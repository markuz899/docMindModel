# Resume here

State at the end of 2026-09-16. Seven commits, clean tree, 119 tests passing,
nothing running.

```bash
cd docmind-model
.venv/bin/python -m pytest tests/ -q          # expect 119 passed
git log --oneline | head -7
```

---

## 0. Moving to the 16 GB machine

**The repository has no git remote.** Copy the whole `docmind-model/` directory,
`.git` included (1.6 MB), or push it somewhere first. Cloning is not an option
until a remote exists.

Then move the artefacts git does not track:

```bash
./scripts/export_bundle.sh ~/Desktop/docmind-model-essentials.tar.gz   # ~700 KB
```

That carries the teacher cache, the v2 dataset and the staged corpus. The cache
is subscription quota already spent -- 53 responses that would otherwise cost
hours to regenerate. Add `--full` to also carry the v1 adapter, the v1 GGUF and
the downloaded base model (~1.8 GB); all three are recreatable, so move them
only if that is cheaper than rebuilding.

On the new machine:

```bash
tar -xzf docmind-model-essentials.tar.gz -C docmind-model/
cd docmind-model
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/doctor.py          # reports which profile fits
.venv/bin/python -m pytest tests/ -q        # expect 119 passed

# only if you did not bundle --full
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir artifacts/base/Qwen2.5-0.5B-Instruct
```

`scripts/doctor.py` checks Python, RAM, disk, every dependency, the accelerator,
both teacher CLIs and their auth, Node, and which artefacts are present -- then
names the training profile that fits the memory it found.

**What changes with 16 GB.** Two constraints that shaped every decision on the
8 GB machine disappear, and both were suspects in the training divergence:

| | 8 GB | 16 GB |
|---|---|---|
| dtype | `float16` (fp32 OOMs) | `float32` |
| max_seq_length | 1536, keeping 79% of examples | 2048, keeping 96% |
| profile | `model.local-0.5b.yaml` | `model.local-0.5b-16gb.yaml` |

Use `configs/model.local-0.5b-16gb.yaml` + `configs/training.v2-16gb.yaml`.
They already carry the lower learning rate and larger batch that section 1
proposes, so the first probe tests all three changes at once.

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

Next step on the 16 GB machine -- probe before committing to a full run:

```bash
.venv/bin/python -u -m src.training.train \
    --model-config configs/model.local-0.5b-16gb.yaml \
    --training-config configs/training.v2-16gb.yaml \
    --dataset-config configs/dataset.real.yaml \
    --max-steps 12 --output-dir artifacts/probe-lr
```

**Success criterion: loss trending below 5.233**, the base model's score on the
same validation set. Anything above that means training is still making the
model worse. Re-measure that baseline first on the new machine, since float32
will shift it -- the snippet that produced 5.233 is in `scripts/doctor.py`'s
sibling command in section 3 of this file, or reuse
`src.training.data.encode_dataset` with the base model and no adapter.

The 16 GB profile changes three things at once (float32, 2048 tokens, lower LR
with a larger batch). That is deliberate for the first probe -- if it converges,
the blocker is gone. Only then vary them one at a time to learn which mattered,
because the answer decides whether the 1.5B run needs the same treatment.

If it still diverges with all three changed, the remaining suspect is the
supervised-token density itself (5-7% against the demo corpus's 18-19%), which
finishing the dataset fixes on its own -- so do section 2 first and retry.

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
    --from artifacts/corpus/real-dataset.jsonl artifacts/corpus/structured.jsonl \
    --out-dir data/real --version 2.0.0
```

**Do not run that yet.** `artifacts/corpus/real-dataset.jsonl` holds 213
candidates; `data/real/` holds 1134, generated by later codex/claude/selfhosted
runs that never wrote back to it. Rebuilding from the corpus as it stands
discards 921 examples. Regenerate the corpus first, or treat `data/real/` as
the source of record and append to the splits directly
(`scripts/import_structured.py --append`).

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
| migration bundle | `scripts/export_bundle.sh` |
| environment check | `scripts/doctor.py` |
| 16 GB profiles | `configs/model.local-0.5b-16gb.yaml`, `configs/training.v2-16gb.yaml` |
| teacher CLI facts and costs | `docs/teacher-cli.md` |
