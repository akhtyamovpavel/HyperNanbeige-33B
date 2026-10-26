# HyperNanbeige-33B: Surgical Mixed-Precision GGUF & Speculative Decoding on a Single 24 GB GPU

**HyperNanbeige-33B** is a zero-retraining, tensor-level mixed-precision GGUF build and `llama-server` deployment recipe for [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) (52 layers, GQA, SwiGLU) designed to run an agentic coding workflow (**OpenCode**, **Claude Code**, **Cline**) on a **single 24 GB GPU (NVIDIA RTX 3090 / RTX 4090)** at **45–87 tok/s** with a **57,344-token Q8_0 KV cache**.

---

## Why Standard Uniform Quantizations Fail on 24 GB VRAM

When fitting a 33B dense model onto a single 24 GB card alongside a 56k–64k context window for agentic coding (which injects ~13k tokens of system prompts, tool schemas, and multi-file diffs), uniform quantization forces a bad compromise:

1. **Uniform `IQ4_XS` (`17.46 GiB`, `~4.25 bpw`):** leaves barely enough VRAM for a 32k–40k context window once CUDA compute buffers (~590 MiB) and embeddings are allocated.
2. **Uniform `IQ2_XXS` (`9.33 GiB`, `~2.06 bpw`) or sparse `v1` (`10.94 GiB`, `~2.28 bpw`):** frees 12+ GiB of VRAM for 98k–131k context, but sub-3-bit quantization in `ffn_down` and middle-layer `attn_q` introduces logit jitter on rare tokens (e.g., Cyrillic string literals or nested `try/except` indentation in multi-step file edits).

### The `HyperNanbeige-33B v2` Solution (`16.71 GiB`, `~3.82 bpw`)

Using zero-dependency GGUF tensor surgery (`build_hyper_gguf.py`), **HyperNanbeige-33B v2** transplants **350 out of 366 tensors** directly from `IQ4_XS` (`Q8_0` / `Q6_K` / `IQ4_XS`) and compresses **only 16 tensors** (`ffn_gate.weight` and `ffn_up.weight` in 8 non-anchor layers: `blk.17..20` and `blk.31..34`) to `IQ2_XXS`:

| Tensor Group | Layers | Quantization Type | Source | Why It Matters |
| :--- | :--- | :--- | :--- | :--- |
| `token_embd`, `output` | Global | `IQ4_XS` / `Q6_K` | `IQ4_XS` | Preserves vocabulary embeddings & final logit head |
| `attn_k`, `attn_v` | All `52/52` (`blk.0..51`) | **`Q8_0`** | `IQ4_XS` | Exact key/value projections for long-context retrieval & tool calls |
| `attn_q`, `attn_output` | All `52/52` (`blk.0..51`) | **`IQ4_XS` / `Q6_K`** | `IQ4_XS` | Full query attention & output projection across every layer |
| `ffn_down` | All `52/52` (`blk.0..51`) | **`IQ4_XS` / `Q6_K`** | `IQ4_XS` | Writes directly into the residual stream; compressing it causes token typos |
| `ffn_gate`, `ffn_up` | `44/52` layers | **`IQ4_XS`** | `IQ4_XS` | Full SwiGLU capacity in 85% of the network |
| `ffn_gate`, `ffn_up` | `8/52` (`blk.17..20, 31..34`) | **`IQ2_XXS`** | `IQ2_XXS` | Saves ~0.75 GiB VRAM in redundant middle-layer expansion with zero quality loss |

**Resulting Tensor Histogram (`366` tensors total, `16.712 GiB`):**
- `IQ4_XS`: 232 tensors
- `F32`: 105 tensors (layer norms)
- `Q8_0`: 104 tensors (`attn_k` and `attn_v` across all 52 layers)
- `IQ2_XXS`: 16 tensors (`ffn_gate` and `ffn_up` in 8 layers)
- `Q6_K`: 9 tensors

---

## Three Critical `llama-server` Optimizations Discovered

### 1. Symmetric `-ctk q8_0 -ctv q8_0` KV Cache (Why Asymmetric `q8_0/q4_0` Kills Speed)

Many guides recommend asymmetric KV cache quantization (`-ctk q8_0 -ctv q4_0`) to save VRAM. However, standard `ggml-cuda` builds compile native FlashAttention vector kernels only for **symmetric** KV types (`_Z33ggml_cuda_flash_attn_ext_vec_caseILi128EL9ggml_type8ELS0_8EEv` for `q8_0/q8_0` and `type2/type2` for `q4_0/q4_0`).

When `-ctk q8_0 -ctv q4_0` is passed, `llama.cpp` falls back to generic `f16` dequantization in FlashAttention, cutting prefill speed in half (`478 tok/s` vs `1100 tok/s`) and hurting decode throughput. Switching to symmetric **`-ctk q8_0 -ctv q8_0`** activates the compiled CUDA kernel and fits **57,344 tokens** comfortably in `23,342 / 24,576 MiB` VRAM on an RTX 3090.

### 2. Draftless N-Gram Speculative Decoding + GPU Backend Sampling

Adding `--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 12 -bs --spec-draft-backend-sampling --cache-reuse 256` requires **0 MB of extra VRAM** (no separate draft model) and accelerates structured code / JSON tool-call generation from `~30 tok/s` to **`45–87 tok/s`** (`draft acceptance = 35%–79%`).

> **Note:** Avoid passing `--reasoning-budget <N>` when using `--spec-draft-backend-sampling`, because `llama-server` implements reasoning budgets via a CPU GBNF grammar sampler (`W common_sampler_init: backend sampling is not compatible with grammar, disabling`), which disables GPU backend sampling during `<think>...</think>`.

### 3. Preventing Context Explosion in Multi-Turn Agents (`--no-reasoning-preserve`)

By default, recent `llama-server` builds enable `--reasoning-preserve`, which re-injects all previous turns' `<think>...</think>` blocks back into the prompt when rendering the Jinja chat template. In a 10-step agentic loop (**OpenCode**), this adds 30,000+ tokens of stale thoughts and triggers `exceed_context_size_error`. Passing **`--no-reasoning-preserve`** keeps the prompt history lean (~37k tokens even after 13+ tool-use iterations) with `f_keep = 0.996` KV cache prefix reuse.

---

## Benchmarks on a Single NVIDIA RTX 3090 (24 GB VRAM)

| Metric | Value (`HyperNanbeige-33B v2`) |
| :--- | :--- |
| **Model File Size** | `16.712 GiB` (`3.82 bpw`) |
| **Unified KV Cache** | `57,344` tokens (`-ctk q8_0 -ctv q8_0`, `-np 4` parallel slots) |
| **Total VRAM Used** | `23,342 MiB / 24,576 MiB` (`1,234 MiB` free safety margin) |
| **Prompt Prefill Speed** | `900 – 1,115 tok/s` (`f_keep = 0.996` slot prefix reuse) |
| **Single-Stream Decode (Reasoning)** | `31 – 34 tok/s` |
| **Single-Stream Decode (Code / JSON / SQL)** | **`45 – 87 tok/s`** (with N-gram speculative decoding) |
| **4-Stream Parallel Aggregate Throughput** | **`54.4 tok/s`** (`n_parallel = 4`) |

---

## Quick Start

### 1. Build `Nanbeige4.3-33B-Hyper.gguf` (or download from Hugging Face)

If you have the base `IQ2_XXS` and `IQ4_XS` GGUF files (e.g. from [`bartowski/Nanbeige_Nanbeige4.3-33B-GGUF`](https://huggingface.co/bartowski/Nanbeige_Nanbeige4.3-33B-GGUF)):

```bash
python3 build_hyper_gguf.py \
  --base /path/to/Nanbeige_Nanbeige4.3-33B-IQ2_XXS.gguf \
  --donor /path/to/Nanbeige_Nanbeige4.3-33B-IQ4_XS.gguf \
  --output ./models/Nanbeige4.3-33B-Hyper.gguf \
  --profile v2-quality
```

### 2. Launch `llama-server`

```bash
python3 run_server.py \
  --model ./models/Nanbeige4.3-33B-Hyper.gguf \
  --llama-server /path/to/llama-server \
  --ctx-size 57344 \
  --kv-type q8_0 \
  --benchmark
```

### 3. Connect OpenCode (`opencode.json`)

Copy [`configs/opencode.json`](configs/opencode.json) into your project directory and run `opencode`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "gpu-server": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local GPU Server (RTX 3090 24GB)",
      "options": {
        "baseURL": "http://127.0.0.1:8091/v1",
        "apiKey": "local-no-key"
      },
      "models": {
        "Nanbeige4.3-33B-Hyper": {
          "name": "Nanbeige4.3-33B-Hyper (Local RTX 3090)",
          "tool_call": true,
          "reasoning": true,
          "temperature": true,
          "limit": {
            "context": 57344,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "gpu-server/Nanbeige4.3-33B-Hyper",
  "small_model": "gpu-server/Nanbeige4.3-33B-Hyper"
}
```

### 4. Train & Build the 4-Layer Neural Speculative Drafter (`Nanbeige4.3-Draft-4L.gguf`)

Because `Nanbeige4.3-33B` uses a `166,912`-token vocabulary, external 0.5B–1.5B models (such as Qwen) cannot act as speculative drafters in `llama-server --model-draft`. We provide a complete **Online Layer-Pruned Distillation** pipeline (`train_draft_nanbeige.py` + `build_draft_gguf.py`):

1. **Online Knowledge Distillation (`train_draft_nanbeige.py`):**
   - Initializes a 4-layer student from anchor layers `[0, 17, 34, 51]` of `Nanbeige/Nanbeige4.3-33B`, sharing the frozen `embed_tokens`, `norm`, and `lm_head` pointers in GPU VRAM (`0 MB` duplication).
   - Trains on agentic coding & `<think>...</think>` traces (`data/nanbeige_agentic_traces.jsonl`) using Multi-Layer Hidden Alignment (`SmoothL1 + Cosine`), Pre-Head RMSNorm Alignment, and Chunked Top-128 Symmetric KL Divergence.
   ```bash
   python3 train_draft_nanbeige.py \
     --teacher Nanbeige/Nanbeige4.3-33B \
     --traces ./data/nanbeige_agentic_traces.jsonl \
     --out-dir ./models/Nanbeige4.3-Draft-4L \
     --layers 0,17,34,51 \
     --epochs 6
   ```

2. **GGUF Stitching & Quantization (`build_draft_gguf.py`):**
   - Applies HuggingFace-to-GGML RoPE head permutation (`56` Q heads, `8` KV heads), quantizes attention weights to `Q8_0` and MLP weights to `Q4_0`, and stitches the 4 distilled layers with the exact vocabulary embeddings and `output.weight` from the base GGUF (`~1.2 GiB` total):
   ```bash
   python3 build_draft_gguf.py \
     --base-gguf ./models/Nanbeige_Nanbeige4.3-33B-IQ2_XXS.gguf \
     --safetensors ./models/Nanbeige4.3-Draft-4L/draft_layers_only.safetensors \
     --out ./models/Nanbeige4.3-33B-Draft-4L.gguf
   ```

## License

MIT License (scripts and configuration). Model weights inherit the license of [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) (Apache 2.0).
