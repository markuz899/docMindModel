# Design decisions and deliberate deviations

## Transformers `Trainer` instead of TRL's `SFTTrainer`

The brief lists TRL as a preferred dependency. We use `transformers.Trainer`
with a ~40-line collator instead (`src/training/data.py`).

Reason: the fine-tune's whole point is that the loss covers the *answer only*.
Doing that through TRL means `DataCollatorForCompletionOnlyLM`, which locates
the response by matching a template string in the decoded text — fragile across
chat templates and tokenizers. Rendering the prompt and the full conversation
through the chat template and masking by token count is exact, and it fails
loudly (`chat template is not prefix-stable`) instead of silently training on
the prompt. It also removes a dependency whose `SFTTrainer` signature has
changed repeatedly across releases.

TRL remains a drop-in alternative if you want its extras (packing, DPO); the
dataset format is unchanged.

## Groundedness is measured on technical tokens

A plain word-overlap groundedness score punishes an Italian answer written over
English documentation, which is the normal case for this project. The quality
gate therefore scores **technical tokens only** — identifiers, routes, table
names, config keys, numbers (`src/text.py: technical_groundedness`) — and the
prose-overlap score is kept as a reported metric with a low floor.

## Bilingual data on purpose

Questions and answers are ~2/3 Italian, ~1/3 English, over English
documentation. This matches the target usage (an Italian-speaking developer
asking about English project docs) and forces the model to ground on the
context rather than on lexical copying. The system prompt instructs the model
to answer in the language of the question.

## Test split = whole unseen projects

With four documentation corpora, an exact 80/10/10 split and group separation
are incompatible: a project is an indivisible ~25% block. The policy is:
one whole project is held out as the test set (the only honest measure of
"works on documentation it has never seen"), validation is a family-disjoint
slice of the remaining projects used for early stopping. Actual sizes are
printed and recorded in `data/generated/dataset_card.json`.

## Train and evaluation refusal templates are disjoint

`src/dataset/negatives.py` keeps two template sets. `TRAIN_TEMPLATES` feeds
the training set, `EVAL_TEMPLATES` feeds `benchmarks/hallucination.jsonl`, and
they share no topic. Measuring refusal with the phrasings the model was trained
on would measure memorisation.

Every generated unanswerable example is also *probe-checked*: if any of the
template's probe terms appears in the sampled context, the template is skipped.
A generated "unanswerable" item can therefore never be accidentally answerable.

## An empty answer counts as a failed refusal

On the hallucination benchmark, a model that outputs nothing has not refused.
It is counted as a failure and `empty_rate` is reported alongside, so a broken
generation setup cannot be mistaken for good behaviour.

## Distractors never collide with real sources

Citations carry no project name, and file names repeat across projects
(`01-overview.md#Purpose` exists in all four). Distractor sampling therefore
skips any chunk whose `(source, heading)` matches one already in the context.

## Quantization is optional, not assumed

`bitsandbytes` has no macOS arm64 wheels. `quantization.enabled: true` on a
machine without a usable bitsandbytes emits a warning and falls back to an
unquantized load; LoRA still trains, it just uses more memory. QLoRA is enabled
by `pip install -r requirements-cuda.txt` on a CUDA box.
