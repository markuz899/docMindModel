# docmind-model

A complete pipeline that **specialises a small open-weight instruct model**
(LoRA / QLoRA supervised fine-tuning) for one job:

> **question + retrieved technical documentation → grounded technical answer**

It does not train a language model from scratch, and it does not teach the model
any particular repository. The behaviour being trained is general:

- answer **only** from the supplied context;
- read software documentation — APIs, classes, services, databases, configs,
  errors, flows;
- separate documented facts from hypotheses;
- say "the documentation does not contain this" instead of inventing;
- cite the sources it used, in the exact `[file — heading]` form;
- ignore irrelevant retrieved sections;
- report contradictions instead of picking a winner.

The project content is supplied in the context at inference time, by the
retriever in the DocMind desktop app.

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Base model](#base-model)
4. [Dataset format](#dataset-format)
5. [The demo dataset](#the-demo-dataset)
6. [Dataset generation from real docs](#dataset-generation-from-real-docs)
7. [Teacher distillation](#teacher-distillation)
8. [Quality gate](#quality-gate)
9. [Splits](#splits)
10. [Baseline evaluation](#baseline-evaluation)
11. [Fine tuning](#fine-tuning)
12. [Evaluation](#evaluation)
13. [Inference](#inference)
14. [Export to GGUF and quantization](#export-to-gguf-and-quantization)
15. [Using it with llama.cpp](#using-it-with-llamacpp)
16. [Using it in the DocMind app](#using-it-in-the-docmind-app)
17. [Versioning](#versioning)
18. [Repository layout](#repository-layout)
19. [Smoke test](#smoke-test)
20. [What has actually been run](#what-has-actually-been-run)

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add -r requirements-cuda.txt for QLoRA

./scripts/smoke_test.sh                  # whole pipeline, ~1 minute, no downloads
```

The smoke test runs the unit tests, builds the dataset and benchmarks, runs the
teacher pipeline with the mock provider, builds a tiny local model, trains it
for 4 steps, then runs inference, evaluation and a base-vs-tuned comparison.
It downloads nothing. Its *answers* are gibberish by design — what it proves is
that every stage executes.

Real run:

```bash
python scripts/build_dataset.py                                   # phase 2-4
python scripts/build_benchmarks.py
python -m src.evaluation.evaluate --out artifacts/eval/baseline_results.json   # phase 5
python -m src.training.train                                      # phase 6
python scripts/evaluate_models.py --adapter artifacts/adapter     # phase 7
python -m src.export.merge --adapter artifacts/adapter            # phase 8
python -m src.export.gguf --test
```

---

## Architecture

```
data/raw/projects/**.md ──┐
                          ├─► chunker (by heading) ─► context bundles ─► teacher LLM ─► candidates
data/raw/seed/*.yaml ─────┘                                                   │
                                                                              ▼
                                                                  quality gate (schema,
                                                                  citations, grounding)
                                                                              │
                          group-aware split ◄────────────────────────────────┘
                                   │
          ┌────────────────────────┼───────────────────────────┐
          ▼                        ▼                           ▼
   data/train/train.jsonl   data/validation/…        data/test/test.jsonl
          │                                          benchmarks/*.jsonl
          ▼                                                  │
   LoRA / QLoRA SFT  ─► artifacts/adapter ──► merge ──► GGUF ──► Q4_K_M
   (answer-only loss)         │                                   │
                              └────────► evaluation ◄─────────────┘
                                   (7 evaluators + 3 benchmarks)
```

Everything that touches the model formats its input through `src/prompt.py`.
There is exactly one implementation of the prompt contract, shared by the
dataset builder, the trainer, the inference engine and the evaluators — if they
ever drifted apart, the fine-tune would silently degrade.

### System prompt

```
You are a technical documentation assistant.
Answer using only the provided project documentation.
Do not use your pretrained knowledge to invent project-specific facts.
Clearly separate documented facts from hypotheses.
If the documentation does not contain enough information, say so explicitly.
When suggesting investigation steps, explain why they follow from the supplied documentation.
Cite the relevant sources using the [file — heading] form shown in the documentation blocks.
Answer in the language of the question.
```

### User message

```
## Project documentation

[04-api-reference.md — GET /me]
Returns the authenticated user. …

[05-main-flows.md — Current User Flow]
1. The client calls GET /me with a bearer token. …

## Question

La rotta /me restituisce dati sbagliati. Da dove dovrei iniziare?
```

The citation token is printed verbatim above each block so the model copies it
instead of reconstructing (and mangling) file names.

---

## Base model

Default: **`Qwen/Qwen2.5-1.5B-Instruct`** — Apache-2.0, not gated,
`Qwen2ForCausalLM`, native in Transformers, standard PEFT target modules,
converts to GGUF and quantizes cleanly.

The full comparison (licenses, gating, GGUF support) and the reasons the 3B
Qwen and Llama 3.2 3B were rejected are in [`docs/model-choice.md`](docs/model-choice.md).

Nothing in the repository hardcodes it. `configs/model.yaml`:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
max_seq_length: 4096
quantization:
  enabled: false          # true + bitsandbytes -> QLoRA 4-bit
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

Shipped profiles:

| Config | Base model | Purpose |
|---|---|---|
| `configs/model.yaml` | Qwen2.5-1.5B-Instruct | default target |
| `configs/model.local-0.5b.yaml` | Qwen2.5-0.5B-Instruct | fast laptop iteration; used for this repo's validation run |
| `configs/model.smoke.yaml` | locally built ~0.3M random model | offline CI smoke test |

---

## Dataset format

One JSON object per line:

```json
{
  "id": "acme-001",
  "question": "La rotta /me restituisce dati sbagliati. Da dove dovrei iniziare?",
  "context": [
    {"source": "04-api-reference.md", "heading": "GET /me", "content": "…"},
    {"source": "05-main-flows.md", "heading": "Current User Flow", "content": "…", "relevant": false}
  ],
  "answer": "…",
  "category": "bug_investigation",
  "difficulty": "hard",
  "contains_code": true,
  "answerable": "full",
  "source_count": 2,
  "project": "acme-identity",
  "relevant_sources": ["04-api-reference.md#GET /me"],
  "must_include": ["ProfileService", "display_name"],
  "must_not_include": ["database timeout"],
  "tags": ["seed", "distractor"],
  "origin": "seed"
}
```

Required: `id`, `question`, `context` (≥1 chunk), `answer`. Everything else is
optional metadata, validated by `src/dataset/schema.py` (pydantic).

| Field | Meaning |
|---|---|
| `category` | one of the twelve question shapes (below) |
| `difficulty` | `easy` / `medium` / `hard` |
| `contains_code` | derived if omitted |
| `answerable` | `full` / `partial` / `none` — how much of the question the context supports |
| `source_count` | derived if omitted |
| `project` | group key for the split — keeps a whole corpus on one side |
| `relevant_sources` | gold `source#heading` citations; drives the relevant-source metric |
| `must_include` | key facts a correct answer has to mention |
| `must_not_include` | plausible facts **absent** from the context; asserting them is a hallucination |
| `tags` | `seed`, `distractor`, `contradiction`, `unanswerable`, `benchmark`, … |
| `origin` | `seed` / `synthetic` / `teacher:<provider>:<model>` |

`context[].relevant: false` marks a deliberately irrelevant retrieved section.

### Categories

`how_it_works`, `route`, `bug_investigation`, `dependencies`, `data_flow`,
`database`, `configuration`, `integrations`, `errors`, `architecture`,
`comparison`, `multi_source`.

---

## The demo dataset

`data/raw/projects/` contains four fictional but realistic documentation
corpora, deliberately different in stack and vocabulary:

| Project | Stack | Domain |
|---|---|---|
| `acme-identity` | Node / TypeScript | accounts, profiles, JWT, `/me` |
| `billing-core` | Python / FastAPI | Stripe, invoices, webhooks, retries |
| `docsync-platform` | Java / Spring | rendering, signing, delivery providers |
| `mesh-gateway` | Go | routing, timeouts, rate limits, Kubernetes |

`data/raw/seed/*.yaml` holds 66 hand-written examples. They reference
documentation by `file.md#Heading`, and `src/dataset/build.py` resolves those
references against the real Markdown — so a seed answer can never drift away
from the documentation it cites. From those, the builder derives:

- **distractor variants** — the same question and answer, re-issued with 3-4
  irrelevant sections from other projects bolted on. More noise in the context
  must not change the answer.
- **generated unanswerable examples** — questions about fact *types* no page in
  the corpus contains (p99 latency, SLA, cost, ownership, license, coverage,
  rollback, on-call, memory). The correct answer is a refusal, by construction.

```
built 184 examples (0 rejected by the quality gate)
  train        127   acme-identity, billing-core, docsync-platform
  validation    16   (family-disjoint slice of the same three)
  test          41   mesh-gateway  (held out entirely)
```

Answerability mix: 75 `full`, 42 `none`, 10 `partial` in train. Roughly a third
of the training set teaches the model **not** to answer — that is not padding,
it is the point.

The three deliberate behaviours worth calling out:

**Negative examples.** The corpus has no latency, cost or ownership data. Asked
for them, the reference answer states exactly what is documented, what is
missing, and what source would be needed — and never guesses a cause.

**Contradictions.** `acme-identity` documents a 15-minute access token TTL in
`06-configuration.md` and a 60-minute one in `08-token-lifetimes.md`;
`mesh-gateway` documents a 30000 ms default upstream timeout in
`04-configuration.md` and 10000 ms in `05-timeouts.md`. The reference answers
report the conflict and cite both sources. They never pick a winner.

**Partial answers.** "How are refresh tokens stored, and how often is the RS256
key rotated?" — the first half is documented, the second is not, and the
reference answer marks the boundary explicitly.

### Benchmarks

| File | Size | What it measures |
|---|---|---|
| `benchmarks/hallucination.jsonl` | 120 | does the model invent an answer when the context has none |
| `benchmarks/retrieval_noise.jsonl` | 58 | relevant sources + exactly 4 irrelevant ones |
| `benchmarks/bug_investigation.jsonl` | 12 | documented path, relevant components, reasonable investigation points, unknowns flagged |

The hallucination benchmark uses question templates **disjoint** from the ones
used in training (`src/dataset/negatives.py`), so it measures the behaviour,
not memorised phrasings.

---

## Dataset generation from real docs

```bash
python scripts/generate_dataset.py --docs /path/to/your/docs --provider mock
```

Pipeline (`src/generation/`):

1. read Markdown recursively; subdirectories of `--docs` become project groups;
2. split each file by heading, ignoring `#` inside fenced code blocks;
3. build context combinations — 1-4 relevant chunks, plus irrelevant chunks from
   other projects at `distractor_ratio`, shuffled;
4. render a teacher prompt with the documentation blocks and the task rules;
5. call the teacher, parse the JSON array (tolerating fences and prose);
6. validate every item through the quality gate;
7. write `data/generated/candidates-<date>.jsonl` and
   `candidates-<date>-rejected.jsonl` with the reason for every rejection.

Tunables live in `configs/dataset.yaml` under `generation:` and can be
overridden on the command line.

---

## Teacher distillation

Providers are behind a two-method interface (`src/generation/providers.py`):

| Provider | Install | Key |
|---|---|---|
| `mock` | — | none; deterministic, offline, used by CI |
| `anthropic` | `pip install anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `pip install openai` | `OPENAI_API_KEY` |

Keys are optional and read from the environment; nothing fails at import time if
they are absent.

```bash
export ANTHROPIC_API_KEY=...
python scripts/generate_dataset.py --docs ./docs --provider anthropic \
    --model claude-opus-5 --max-contexts 200 --questions-per-context 2
```

The teacher is asked for `question`, `answer`, `category`, `difficulty`,
`answerability`, `relevant_sources` and `must_include` — and is explicitly told
to produce unanswerable questions for a share of the contexts, to ignore the
irrelevant blocks, and to report contradictions rather than resolve them.

**The output is candidate data.** Review it, then validate and merge:

```bash
python scripts/validate_dataset.py data/generated/candidates-2026-09-16.jsonl \
    --fix-out data/generated/reviewed.jsonl
```

---

## Quality gate

`src/dataset/quality.py`. An example is rejected when:

- the question or answer is empty, too short or too long;
- the answer is long **and** disproportionate to its context (rambling);
- it cites a **source that is not in the context** (fabricated citation);
- it cites a heading that does not exist under a real source (misattribution);
- it names an identifier — class, route, table, config key — that appears
  nowhere in the context;
- technical grounding falls below the floor;
- `answerable: none` but the answer does not state the gap;
- `answerable: partial` but the answer never flags the unknown part;
- a contradiction example does not name the conflict or cite both sources;
- the answer asserts something listed in `must_not_include`;
- `relevant_sources` points outside the context;
- the id is a duplicate.

Grounding is scored on **technical tokens** (identifiers, routes, numbers,
config keys), not on prose overlap — see [`docs/decisions.md`](docs/decisions.md).
A refusal gets a small identifier budget, because naming the missing topic is
the whole point of a refusal.

```bash
python scripts/validate_dataset.py --all      # non-zero exit on any failure
```

---

## Splits

Naive per-example splitting leaks: the same section, reworded or re-issued with
extra distractors, lands on both sides and the test score measures memorisation.

Policy (`configs/dataset.yaml` → `split.group_by: project`):

- **test** = whole projects, held out entirely — the only honest measure of
  "works on documentation it has never seen";
- **train / validation** = the remaining projects, split by *family* so an
  example and its distractor variants never straddle the boundary;
- near-duplicate removal (token Jaccard ≥ 0.9) on top.

The requested 80/10/10 is a target, not a guarantee: a project is an
indivisible block. Achieved sizes are printed and recorded in
`data/generated/dataset_card.json`.

---

## Baseline evaluation

Measure the base model **before** touching it, with the same suites used after:

```bash
python -m src.evaluation.evaluate --out artifacts/eval/baseline_results.json
```

Writes the summary JSON plus `baseline_results-predictions.jsonl` with every
generated answer, so regressions can be read, not just plotted.

---

## Fine tuning

```bash
python -m src.training.train                        # configs/training.yaml
python -m src.training.train --resume latest        # resume from a checkpoint
python -m src.training.train --smoke                # 4 steps, tiny local model
```

LoRA, or QLoRA when `quantization.enabled: true` and bitsandbytes is usable.
Everything is configurable in `configs/training.yaml`: rank, alpha, dropout,
learning rate, epochs, batch size, gradient accumulation, sequence length,
warmup, weight decay, scheduler, optimizer, checkpoint strategy.

**The loss covers the answer only.** The prompt — system message, documentation
blocks, question — is masked out with `-100`. Training on the prompt would
teach the model to reproduce documentation blocks, which is the opposite of the
goal. The masking is done by rendering both the prompt and the full
conversation through the chat template and masking by token count; if the
template is not prefix-stable the trainer fails loudly instead of silently
training on the prompt.

Memory behaviour, by default:

- LoRA adapters only (base weights frozen);
- gradient accumulation (effective batch 16 from a per-device batch of 1);
- gradient checkpointing on;
- mixed precision resolved per device — bf16 on Ampere+, fp16 on older CUDA,
  fp32 on MPS and CPU (fp16 training on MPS produces NaNs);
- 4-bit base weights when QLoRA is available.

Nothing assumes a datacentre GPU. On Apple Silicon the pipeline runs as-is;
for a full 1.5B run a consumer GPU or a rented box is faster, and the adapter
is a few tens of MB either way.

Outputs to `artifacts/adapter/`: the adapter, the tokenizer, and
`training_metrics.json` with hyperparameters, token statistics and losses.

---

## Evaluation

Seven evaluators, all lexical and deterministic — no judge model, no network,
identical numbers on a laptop and in CI (`src/evaluation/metrics.py`):

| # | Metric | What it is |
|---|---|---|
| 1 | `answer_correctness` | token-F1 against the reference, blended with recall of `must_include` key facts |
| 2 | `groundedness` | share of the answer's technical tokens present in the context |
| 3 | `citation_f1` | precision/recall of `[file — heading]` citations against the context and the gold sources |
| 4 | `refusal_correct_rate` | refuses when it must, answers when it can |
| 5 | `hallucinated_rate` | **headline metric** — fabricated citation, invented identifier, forbidden fact, or failing to refuse |
| 6 | `relevant_source_usage` | share of citations that hit gold sources; `distractor_citations` counts the rest |
| 7 | `conciseness` | length against the reference, penalised above 1.5× |

```bash
python -m src.evaluation.evaluate --adapter artifacts/adapter \
    --out artifacts/eval/finetuned_results.json

python scripts/evaluate_models.py --adapter artifacts/adapter   # side-by-side table
```

`evaluate_models.py` loads one model at a time, so it runs on a machine that can
hold exactly one of them, and prints base / fine-tuned / delta per suite. Pass
`--base-results artifacts/eval/baseline_results.json` to reuse an earlier
baseline instead of regenerating it.

Generation is the expensive half; scoring is not. When a metric definition
changes, rescore the stored answers rather than regenerating them:

```bash
python scripts/rescore.py artifacts/eval/baseline_results.json
```

### Measured results

`Qwen/Qwen2.5-0.5B-Instruct` before and after 3 epochs of LoRA on the demo
dataset (920 s on an 8 GB M1), identical suites and decoding. **26 of 28
metric comparisons improve.**

| suite | n | hallucination ↓ | refusal ↑ | correctness ↑ | citation F1 ↑ |
|---|---|---|---|---|---|
| `test` (unseen project) | 41 | 0.415 → **0.049** | 0.585 → 0.902 | 0.177 → 0.474 | 0.024 → **0.616** |
| `hallucination` | 120 | 0.942 → **0.017** | 0.058 → **0.983** | 0.124 → 0.507 | 0.075 → **0.983** |
| `retrieval_noise` | 58 | 0.103 → 0.207 ✗ | 0.879 → 0.776 ✗ | 0.235 → 0.372 | 0.000 → 0.358 |
| `bug_investigation` | 12 | 0.417 → **0.083** | 0.000 → **1.000** | 0.137 → 0.167 | 0.000 → 0.056 |

On 120 questions the documentation cannot answer, the base model invents an
answer 94% of the time and the tuned model 1.7%. It also learned to cite:
citation F1 went from ~0 to 0.62 on a project it never saw in training.

The two ✗ cells are one real regression — **the model now over-refuses**
(`false_refusal_rate` on answerable questions: 0.00 → 0.17). It memorised the
three refusal templates that generate a third of the training set.
[`RESULTS.md`](RESULTS.md) shows the failing answer, the diagnosis, and a
second training run that tested the obvious fix and **made things worse** —
diversifying the refusal phrasings without adding examples dropped refusal
from 0.98 to 0.47 *and* raised false refusals from 0.17 to 0.30. Variety
without volume destroys the signal. That is the reason the suite has seven
evaluators rather than one, and the reason the teacher pipeline exists.

A model that outputs nothing has not refused: empty answers count as failures on
the hallucination benchmark and `empty_rate` is reported next to them.

---

## Inference

```bash
python -m src.inference.chat --context docs/ --question "Come funziona GET /me?"
python -m src.inference.chat --adapter artifacts/adapter --context context.json
python -m src.inference.chat --context data/test/test.jsonl --show-prompt
```

`--context` accepts a Markdown file, a directory of Markdown, a JSON list of
`{source, heading, content}` chunks, or a JSONL dataset file. Without
`--question` it drops into an interactive prompt.

The model answers in natural prose — no forced JSON — but in a consistent shape:
direct answer, documented facts with citations, then hypotheses and unknowns
clearly separated.

---

## Export to GGUF and quantization

```bash
python -m src.export.merge --adapter artifacts/adapter     # LoRA -> standalone weights
python -m src.export.gguf --test                            # GGUF + quantizations + smoke prompts
```

The merge always runs unquantized on CPU — merging an adapter into 4-bit weights
loses its precision.

`src/export/gguf.py` finds llama.cpp via `--llama-cpp`, `$LLAMA_CPP_DIR`,
`llama-quantize` on `PATH`, `./vendor/llama.cpp`, or `~/llama.cpp`. If it finds
nothing it prints the exact setup commands and exits 2 — it never pretends the
export happened.

Quantizations are configured in `configs/model.yaml`:

```yaml
export:
  quantizations: [Q4_K_M, Q5_K_M, Q8_0]
  primary_quantization: Q4_K_M     # the build shipped with the app
```

With `--test`, real prompts from `data/test/test.jsonl` are run through the
quantized model with `llama-cli` and the answers are printed and recorded in
`model_metadata.json`.

---

## Using it with llama.cpp

```bash
llama-cli -m artifacts/export/gguf/merged-Q4_K_M.gguf \
  --temp 0 -c 4096 -n 512 -no-cnv -f prompt.txt
```

`prompt.txt` in ChatML, exactly as the model was trained:

```
<|im_start|>system
You are a technical documentation assistant.
…
<|im_end|>
<|im_start|>user
## Project documentation

[04-api-reference.md — GET /me]
…

## Question

Come funziona GET /me?<|im_end|>
<|im_start|>assistant
```

As a local server:

```bash
llama-server -m artifacts/export/gguf/merged-Q4_K_M.gguf -c 4096 --port 8080
```

and POST OpenAI-style chat messages to `/v1/chat/completions` with the same
system prompt. `src/prompt.py` renders the user message; reuse it rather than
reimplementing the format.

---

## Using it in the DocMind app

The app owns retrieval; this model owns answering. The contract between them:

1. the app retrieves N documentation sections for the question;
2. it renders them with `build_user_message(question, chunks)` from
   `src/prompt.py` — each chunk is `{source, heading, content}`;
3. it prepends `SYSTEM_PROMPT`;
4. it runs the `Q4_K_M` GGUF through llama.cpp (embedded, or `llama-server`);
5. it parses `[file — heading]` citations out of the answer to render source
   links back into the documentation.

Practical notes:

- **Retrieval may be noisy.** The model is trained on contexts that include
  irrelevant sections; it is not trained to handle an *empty* context. Always
  pass at least one chunk.
- **Keep `source` stable.** It is the citation key the app links on.
- **Greedy decoding** (`temperature: 0`) is what the evaluation uses; sampling
  increases hallucination.
- **Context budget.** `max_seq_length: 4096` covers roughly 8-10 documentation
  sections plus the answer. Retrieve fewer, better sections rather than more.
- The model will sometimes answer "the documentation does not contain this".
  That is the trained behaviour, and it is what makes the rest trustworthy —
  surface it as a real answer, not as an error.

---

## Versioning

`artifacts/export/*/model_metadata.json`:

```json
{
  "name": "docmind-model",
  "version": "0.1.0",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_type": "lora",
  "training_dataset": "data/train/train.jsonl",
  "dataset_version": "0.1.0",
  "training_date": "2026-09-16T…",
  "metrics": { "train_loss": …, "eval_loss": … },
  "quantizations": { "Q4_K_M": { "path": "…", "size_mb": … } },
  "gguf_smoke_test": [ … ]
}
```

Dataset provenance is separate, in `data/generated/dataset_card.json`: split
policy, quality thresholds, per-category and per-project counts, and how many
examples came from seeds, augmentation and the teacher.

---

## Repository layout

```
configs/         model.yaml, training.yaml, dataset.yaml (+ smoke / 0.5B profiles)
data/raw/        projects/   four Markdown documentation corpora
                 seed/       hand-written examples, referencing docs by heading
                 benchmarks/ the bug-investigation rubric
data/generated/  built dataset + dataset_card.json + teacher candidates
data/{train,validation,test}/
benchmarks/      hallucination / retrieval_noise / bug_investigation JSONL
src/
  config.py      YAML -> validated pydantic config
  prompt.py      the single prompt contract
  text.py        lexical helpers shared by the quality gate and the evaluators
  runtime.py     device / dtype / quantization / model loading
  dataset/       schema, io, quality, split, build, negatives
  generation/    markdown chunker, teacher providers, distillation pipeline
  training/      answer-only encoding + collator, LoRA/QLoRA trainer
  evaluation/    the seven evaluators, suite runner, CLI
  inference/     engine + chat CLI
  export/        LoRA merge, GGUF conversion + quantization + smoke test
scripts/         build_dataset, build_benchmarks, generate_dataset,
                 validate_dataset, evaluate_models, make_tiny_model, smoke_test.sh
tests/           unit tests over schema, quality, split, chunker, teacher, metrics
notebooks/       dataset inspection
docs/            model-choice.md, decisions.md
```

---

## Smoke test

```bash
./scripts/smoke_test.sh
```

Eight stages, no downloads, ~1 minute: unit tests → dataset build → benchmark
build → dataset validation → teacher pipeline (mock) → tiny model → training
(4 steps) → inference, evaluation and comparison.

### What has actually been run

[`RESULTS.md`](RESULTS.md) is the ledger of what actually executed: all nine
phases on real weights on an 8 GB M1 — baseline, training, evaluation, merge,
GGUF, quantization — with the three defects that had to be fixed to get there
(fp16 for the vocabulary-sized loss tensor, dropping rather than truncating
over-long examples, and releasing the MPS cache every step: 58 s → 9.5 s per
step). The only stage it does not cover is a GGUF prompt run through
llama.cpp, because the installed binaries are x86_64 under Rosetta.

For a real-but-cheap end-to-end run on actual weights:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
    --local-dir artifacts/base/Qwen2.5-0.5B-Instruct
python -m src.training.train --model-config configs/model.local-0.5b.yaml \
                             --training-config configs/training.local-0.5b.yaml
python scripts/evaluate_models.py --model-config configs/model.local-0.5b.yaml \
                                  --adapter artifacts/adapter-0.5b
python -m src.export.merge --model-config configs/model.local-0.5b.yaml \
                           --adapter artifacts/adapter-0.5b
python -m src.export.gguf --model-config configs/model.local-0.5b.yaml --test
```

`src/export/gguf.py` needs `pip install sentencepiece gguf` for llama.cpp's
conversion script, and on Apple Silicon it warns if the llama.cpp binaries are
x86_64 (running under Rosetta, without Metal).

## License

Apache-2.0, matching the default base model. The fictional documentation in
`data/raw/projects/` is part of this repository and describes no real system.
