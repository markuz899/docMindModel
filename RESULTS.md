# Validation run — what was actually executed

Dated 2026-09-16. Everything below was run in this repository; nothing is
projected or estimated. Where a stage could not be completed on this machine,
it says so and why.

## Machine

| | |
|---|---|
| Hardware | Apple M1, **8.6 GB** unified memory |
| OS / Python | macOS 14.0, Python 3.10.4 |
| torch / transformers / peft | 2.14.0 / 5.17.0 / 0.21.0 |
| Accelerator | MPS (`torch.backends.mps.is_available() == True`) |
| llama.cpp | brew build 10566 — **x86_64 binaries under Rosetta** (no Metal) |

## Stage by stage

| Phase | Stage | Status | Evidence |
|---|---|---|---|
| 1 | repo + configs | done | `configs/*.yaml` load through `src/config.py` |
| 2 | dataset schema | done | 52 unit tests pass |
| 3 | dataset generator | done | `scripts/generate_dataset.py --provider mock` → 6 accepted / 0 rejected |
| 4 | demo dataset | done | 184 examples, 0 rejected by the quality gate |
| 5 | **baseline evaluation on real weights** | **done** | `artifacts/eval/baseline_results.json`, Qwen2.5-0.5B-Instruct, 231 generations on MPS |
| 6 | **LoRA training on real weights** | **done** | `artifacts/adapter-0.5b/`, 75 steps, 920 s, train_loss 0.81 / eval_loss 0.70 |
| 7 | **evaluation, base vs fine-tuned** | **done** | `artifacts/eval/comparison.json` — see [Results](#results) |
| 8 | **GGUF export + quantization of the fine-tuned model** | **done** | `artifacts/export/gguf-0.5b/merged-0.5b-Q4_K_M.gguf`, 397.8 MB |
| 8b | GGUF prompt test through llama.cpp | **timed out on this machine** | see [llama.cpp](#llamacpp-on-this-machine) |
| 9 | documentation | done | `README.md`, `docs/`, this file |

## Dataset

```
built 184 examples (0 rejected by the quality gate)
  train        127  acme-identity, billing-core, docsync-platform
  validation    16  family-disjoint slice of the same three
  test          41  mesh-gateway (held out entirely)

benchmarks/hallucination.jsonl      120   (eval-only templates, disjoint from training)
benchmarks/retrieval_noise.jsonl     58   (each with exactly 4 irrelevant sources added)
benchmarks/bug_investigation.jsonl   12   (hand-written rubric)
```

`scripts/validate_dataset.py --all` → **all files valid**.

Sanity check on the evaluators: scoring every reference answer against itself
("oracle") gives `hallucinated_rate = 0.000`, `refusal_correct_rate = 1.000`
and `answer_correctness > 0.95` on all four suites. That is asserted by
`tests/test_evaluation.py::test_oracle_scores_high_on_every_shipped_suite`.

## Baseline

`Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, 256 new tokens, batch 4, MPS.
231 generations in ~20 minutes.

| suite | n | hallucination ↓ | refusal ↑ | correctness ↑ | groundedness ↑ | citation F1 ↑ | rel. sources ↑ | concise ↑ |
|---|---|---|---|---|---|---|---|---|
| `test` | 41 | 0.415 | 0.585 | 0.177 | 0.797 | 0.024 | 0.024 | 0.938 |
| `hallucination` | 120 | **0.942** | 0.058 | 0.124 | 0.869 | 0.075 | 0.058 | 0.895 |
| `retrieval_noise` | 58 | 0.103 | 0.879 | 0.235 | 0.932 | 0.000 | 0.017 | 0.942 |
| `bug_investigation` | 12 | 0.417 | 0.000 | 0.137 | 0.797 | 0.000 | 0.000 | 0.952 |

Read: the untuned model **invents an answer 94% of the time when the context
cannot answer the question**, essentially never cites a source, and never
flags an unknown in a debugging scenario. That is the gap the fine-tune exists
to close, and it is measured, not assumed.

A representative baseline answer — the question was *"Quanto costa in
infrastruttura mesh-gateway?"*, which no supplied section can answer:

> A infraestrutura do Gateway Mesh é composta por vários componentes, incluindo:
> 1. **Gateway**: É o principal componente … 4. **Load Balancer**: …

Wrong language, invented components, no refusal, no citation. Every generated
answer is in `artifacts/eval/baseline_results-predictions.jsonl`.

## Training

`Qwen/Qwen2.5-0.5B-Instruct` + LoRA (r=16, alpha=32, all seven projections),
fp16 on MPS, gradient checkpointing, effective batch 4, 3 epochs, 75 optimizer
steps, **920 seconds**.

```
[data] dropped 30/127 example(s) longer than max_seq_length=640
[train] train tokens: {'examples': 97, 'skipped_overlong': 30, 'truncated': 0,
                       'max_tokens': 632, 'mean_tokens': 442.9, 'trainable_tokens': 11311}
loss 1.936 -> 1.633 -> 1.238 -> 1.047 -> ... -> 0.352 -> 0.381
train_loss 0.8125   eval_loss 0.6988
```

Three things had to be fixed before this ran at all, all of them now part of
the pipeline:

1. **fp32 does not fit.** The dominant tensor is the loss input,
   `seq_len × vocab_size`; Qwen2.5's vocabulary is 151936 tokens, so 970 tokens
   in fp32 is ~590 MB before the backward pass. `dtype: float16` halves it.
   The OOM now prints the remediation list instead of a traceback.
2. **Truncating over-long examples was a correctness bug.** The encoder trimmed
   the prompt from the left, which can delete the very section the answer
   cites — training the model to cite sources it cannot see. `skip_overlong`
   now *drops* those examples (30 of 127 at a 640-token cap) and reports it.
   `TrainingConfig.max_seq_length` caps training only, so the served context
   budget is unaffected.
3. **The MPS allocator grows without bound.** With
   `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` set to dodge the OOM, PyTorch never
   released cached blocks and step time degraded monotonically —
   **33 s → 47 s → 58 s** and climbing, with the process spending 70% of its
   time blocked on swap. An `EmptyCacheCallback` that drops the cache every
   step, plus restoring the default watermark, took the step time to a flat
   **9.5 s**: the same run went from a 2h19m projection to 15m20s actual.

## Results

Base vs fine-tuned, identical suites, identical decoding (greedy, 256 new
tokens, batch 4). **26 of 28 metric comparisons improve.**

| suite | metric | base | fine-tuned | Δ |
|---|---|---|---|---|
| `hallucination` (120) | hallucination ↓ | 0.942 | **0.017** | −0.925 |
| | refusal correct ↑ | 0.058 | **0.983** | +0.925 |
| | citation F1 ↑ | 0.075 | **0.983** | +0.908 |
| | correctness ↑ | 0.124 | 0.507 | +0.383 |
| `test` (41, unseen project) | hallucination ↓ | 0.415 | **0.049** | −0.366 |
| | refusal correct ↑ | 0.585 | 0.902 | +0.317 |
| | correctness ↑ | 0.177 | 0.474 | +0.297 |
| | citation F1 ↑ | 0.024 | **0.616** | +0.591 |
| | relevant sources ↑ | 0.024 | **0.939** | +0.915 |
| | groundedness ↑ | 0.797 | 0.906 | +0.109 |
| `retrieval_noise` (58) | correctness ↑ | 0.235 | 0.372 | +0.136 |
| | relevant sources ↑ | 0.017 | **0.750** | +0.733 |
| | citation F1 ↑ | 0.000 | 0.358 | +0.358 |
| | hallucination ↓ | 0.103 | 0.207 | **+0.104** ✗ |
| | refusal correct ↑ | 0.879 | 0.776 | **−0.103** ✗ |
| `bug_investigation` (12) | refusal correct ↑ | 0.000 | **1.000** | +1.000 |
| | relevant sources ↑ | 0.000 | **1.000** | +1.000 |
| | hallucination ↓ | 0.417 | 0.083 | −0.333 |

The headline result: on 120 questions the supplied documentation cannot
answer, the base model invents an answer 94% of the time and the fine-tuned
model 1.7%. It also learned to cite: citation F1 went from ~0 to 0.62 on a
project it has never seen.

A refusal it produces now:

> **Q:** How often is the data of Authentication flow backed up?
> **A:** The provided documentation does not contain this information. It states
> that every protected route passes through `authMiddleware`, but there is no
> mention of backup frequency or any other such details. Answering this would
> require the configuration file or the backup policy.

### The regression, and what causes it

The two ✗ rows are one behaviour: **the model now over-refuses**.
`false_refusal_rate` on answerable questions went from 0.00 → 0.17 on `test`
and 0.02 → 0.19 on `retrieval_noise`. A clear case from the held-out project:

> **Q:** Quali upstream sono raggiungibili tramite il gateway?
> *(context contains Identity routes, Billing routes and Document routes — it
> answers the question completely)*
> **A:** Le sezioni fornite non sono sufficienti: la documentazione non descrive
> il gateway. Sono presenti le sezioni «Document routes», «Identity routes» e
> «Billing routes»; non c'è alcun riferimento a questo dato per il gateway.

That is `REFUSAL_TEMPLATES_IT[2]` from `src/dataset/negatives.py`, reproduced
almost verbatim, guillemets and all. The model did not learn *when* to refuse;
on unfamiliar-looking context it learned to **emit the refusal template**.

The cause is in the training mix, not in the model: 42 of 127 training
examples (33%) are refusals generated from **three** phrasings per language.
That is enough signal to teach refusal and enough repetition to teach the
form. Concretely, for the next iteration:

1. Drop the refusal share from ~33% to ~20% (`--negatives-per-project`).
2. Generate refusals with a teacher instead of templates
   (`scripts/generate_dataset.py --provider anthropic`) so no two read alike —
   the pipeline for this already exists and is the reason it exists.
3. Add answerable examples over the same corpora to rebalance.
4. Track `false_refusal_rate_answerable` as a gate, not just a reported number:
   a refusal model that refuses everything scores perfectly on the
   hallucination benchmark and is useless.

Point 4 is why the suite has seven evaluators and not one.

### The over-refusal experiment (a measured failure)

Hypothesis 2 above was tested, on the same machine, and it **made the model
worse**. Recorded here because the negative result is the useful part.

One variable changed: the refusal templates went from 3 near-identical
phrasings per language to 8 structurally different ones (distinct refusal
openings in the training set: 3 → 36 of 42). Same corpora, same ratios, same
33% refusal share, same test set, same hyperparameters, same 75 steps.

| suite / metric | base | v1 (3 phrasings) | v2 (36 phrasings) |
|---|---|---|---|
| `hallucination` — hallucination ↓ | 0.933 | **0.017** | 0.533 |
| `hallucination` — refusal correct ↑ | 0.067 | **0.983** | 0.467 |
| `hallucination` — citation F1 ↑ | 0.083 | **0.983** | 0.525 |
| `test` — hallucination ↓ | 0.400 | **0.049** | 0.175 |
| `test` — **false refusal** ↓ | 0.000 | **0.174** | 0.304 |
| `test` — correctness ↑ | 0.172 | **0.474** | 0.341 |
| `retrieval_noise` — false refusal ↓ | 0.019 | **0.192** | 0.308 |
| `bug_investigation` — correctness ↑ | 0.137 | 0.167 | **0.253** |

v2 does not just refuse less when it should — it also refuses **more** when it
should not. Both directions got worse at once, which is the signature of having
learned nothing coherent about when to refuse, rather than of having learned a
different rule.

The arithmetic explains it. 42 refusal examples over 3 phrasings is ~14
reinforcements per surface form, enough for a 0.5B with LoRA r=16 to learn the
behaviour (over-applied, but learned). The same 42 over 36 phrasings is ~1.2
each, below the threshold at which anything is learned. **Variety without
volume destroys the signal instead of generalising it.**

So the diagnosis of over-refusal was wrong in its remedy. The problem is not
too few phrasings, it is too few examples; adding variety to a fixed budget
makes it worse. The fix is variety *and* volume together — which is what a
teacher produces and what templates cannot
(`scripts/generate_dataset.py --provider anthropic`), and is the reason that
pipeline exists.

`src/dataset/negatives.py` is therefore back to the 3-phrasing set, which
reproduces the shipped adapter, with the measurement recorded at the top of the
file so nobody "improves" it again without re-running the benchmark. The
extra refusal markers this experiment surfaced in `src/text.py` were kept: the
detector was genuinely missing legitimate phrasings such as "the documentation
is silent on X", and a guard test now asserts every shipped template is
recognised as a refusal.

## GGUF export

Ran for real on the **fine-tuned** model: LoRA merged into the base weights
(fp16, on CPU), converted and quantized.

```
[merge] wrote artifacts/export/merged-0.5b (988 MB of weights)
[gguf] converting artifacts/export/merged-0.5b -> f16
[gguf] quantizing Q4_K_M ... Q8_0

    Q4_K_M     397.8 MB  merged-0.5b-Q4_K_M.gguf
      Q8_0     531.1 MB  merged-0.5b-Q8_0.gguf
```

`artifacts/export/gguf-0.5b/model_metadata.json` carries the full provenance:
name, version, base model, adapter, dataset version, training date, training
metrics and the size of every quantization.

The conversion tooling is vendored by sparse checkout in `vendor/llama.cpp`
(`convert_hf_to_gguf.py` + `conversion/` + `gguf-py`, ~3 MB). `convert_hf_to_gguf.py`
additionally needs `pip install sentencepiece gguf`; this is noted in
`requirements.txt`.

## llama.cpp on this machine

`llama-cli` from Homebrew here is an **x86_64 binary on an arm64 M1**:

```
$ lipo -archs /usr/local/bin/llama-cli
x86_64
```

It runs under Rosetta with no Metal backend, and generating even 24 tokens from
the 398 MB `Q4_K_M` model did not finish within 150 s under the memory pressure
described above. `src/export/gguf.py` now detects this and says so:

```
WARNING: /usr/local/bin/llama-cli is x86_64 only. On Apple Silicon it runs
under Rosetta, without Metal. Build llama.cpp from source, or install an
arm64 Homebrew (/opt/homebrew), before trusting any timing.
```

Prompt timeouts are recorded as timeouts in `model_metadata.json` rather than
crashing the export. To get a real GGUF prompt test on this machine, build
llama.cpp natively:

```bash
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp-native
cmake -B vendor/llama.cpp-native/build -S vendor/llama.cpp-native -DGGML_METAL=ON
cmake --build vendor/llama.cpp-native/build --config Release -j
python -m src.export.gguf --llama-cpp vendor/llama.cpp-native --test
```

## Summary

The complete pipeline ran on real weights: baseline → training → evaluation →
merge → GGUF → quantization, all nine phases, on an 8 GB M1. The fine-tune cut
the hallucination rate from 94.2% to 1.7% and took citation F1 from ~0 to 0.62
on a documentation corpus it had never seen, at the cost of a measured
over-refusal regression that the evaluation suite caught and that has a
concrete fix.

The one thing not demonstrated here is a GGUF prompt run through llama.cpp,
because the installed binaries are x86_64 under Rosetta.

`./scripts/smoke_test.sh` runs the whole loop offline in about a minute on a
tiny local model, for CI and for checking a change did not break a stage.
