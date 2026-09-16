# Base model selection

Checked on 2026-09-16 against the Hugging Face model API (license, gating,
architecture) and against llama.cpp's `convert_hf_to_gguf.py` model registry.

| Model | Params | License | Gated | Architecture | GGUF | Verdict |
|---|---|---|---|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** | 1.54 B | **apache-2.0** | no | `Qwen2ForCausalLM` | yes | **selected** |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.54 B | apache-2.0 | no | `Qwen2ForCausalLM` | yes | alternative for code-heavy docs |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7 B | apache-2.0 | no | `LlamaForCausalLM` | yes | alternative, weaker on technical text |
| Qwen/Qwen2.5-3B-Instruct | 3.1 B | `other` (Qwen Research) | no | `Qwen2ForCausalLM` | yes | better quality, **non-permissive license** |
| meta-llama/Llama-3.2-3B-Instruct | 3.2 B | `llama3.2` | **manual approval** | `LlamaForCausalLM` | yes | gated + license restrictions |

## Why Qwen2.5-1.5B-Instruct

- **License.** Apache-2.0, not gated. A model shipped inside a desktop
  application cannot depend on a research-only license or on per-user approval.
  This is what rules out the 3B Qwen and Llama 3.2, which are otherwise better.
- **Transformers support.** `Qwen2ForCausalLM` is native in Transformers, so
  `trust_remote_code` stays `false`.
- **PEFT support.** Standard `q/k/v/o_proj` + `gate/up/down_proj` projections,
  the LoRA target set in `configs/model.yaml`.
- **GGUF + quantization.** Qwen2 is in llama.cpp's conversion registry and
  quantizes cleanly to `Q4_K_M`. Verified end to end in this repo with the 0.5B
  sibling (see below).
- **Task fit.** Strong instruction following for its size, heavy code and
  technical-English pretraining, and a stable ChatML template — which matters,
  because the completion-only loss masking depends on the chat template being
  prefix-stable.

## Size target

1.5B is the lower end of the requested 1.5B-3B band, chosen because the model
runs alongside a desktop app rather than on dedicated hardware. Rough
`Q4_K_M` footprints:

| Model | `Q4_K_M` size | Realistic on |
|---|---|---|
| 0.5B | ~0.4 GB | any CPU |
| 1.5B | ~1.1 GB | modern CPU, Apple Silicon, any consumer GPU |
| 3B | ~2.0 GB | Apple Silicon, consumer GPU; sluggish on CPU |

`configs/model.yaml` is the only place the choice is encoded. Switching to any
of the alternatives above is a one-line change plus, for non-Qwen
architectures, a check of `lora.target_modules`.

## Validation run

The end-to-end validation in this repo was executed with
`configs/model.local-0.5b.yaml` (Qwen2.5-0.5B-Instruct, same architecture,
same Apache-2.0 license) so that a full train -> evaluate -> merge -> GGUF ->
llama.cpp cycle completes in minutes on a laptop. Nothing in the pipeline is
specific to either size.
