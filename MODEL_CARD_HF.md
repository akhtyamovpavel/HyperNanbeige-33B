---
license: apache-2.0
base_model:
  - Nanbeige/Nanbeige4.3-33B
  - bartowski/Nanbeige_Nanbeige4.3-33B-GGUF
tags:
  - gguf
  - llama.cpp
  - nanbeige
  - mixed-precision
  - speculative-decoding
  - rtx-3090
  - opencode
  - agentic-coding
pipeline_tag: text-generation
quantized_by: akhtyamovpavel
---

# Nanbeige4.3-33B-Hyper-GGUF (v2, 3.82 bpw Surgical Mixed-Precision for 24 GB GPUs)

**`Nanbeige4.3-33B-Hyper.gguf`** (`16.71 GiB`, `~3.82 bpw`) is a surgical tensor-level mixed-precision GGUF quantization of [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) engineered specifically to run multi-step agentic coding (**OpenCode**, **Claude Code**, **Cline**) on a **single 24 GB GPU (RTX 3090 / RTX 4090)** with a **57,344-token symmetric `Q8_0` KV cache** at **45–87 tok/s**.

- **GitHub Repository (Builder Script, Server Launcher & OpenCode Config):** [https://github.com/akhtyamovpavel/HyperNanbeige-33B](https://github.com/akhtyamovpavel/HyperNanbeige-33B)

---

## Why `HyperNanbeige-33B v2`?

On a 24 GB GPU, standard uniform quantizations of a 33B model force an uncomfortable trade-off when running long-context coding agents:
- Full `IQ4_XS` (`17.46 GiB`) leaves too little VRAM for a 57k `Q8_0` KV cache + CUDA compute buffers.
- Aggressive `IQ2_XXS` (`9.33 GiB`) or sparse mixed builds (`~2.28 bpw`) free VRAM but suffer from sub-3-bit logit jitter on Cyrillic literals and complex code edits when `ffn_down` and middle-layer `attn_q` are compressed.

**`Nanbeige4.3-33B-Hyper.gguf` (`16.712 GiB`)** resolves this by keeping **350 out of 366 tensors** at high precision (`Q8_0` / `Q6_K` / `IQ4_XS`) and compressing **only 16 tensors** (`ffn_gate.weight` and `ffn_up.weight` in 8 non-anchor layers: `blk.17..20` and `blk.31..34`) to `IQ2_XXS`:

| Tensor Group | Layers | Precision | Count |
| :--- | :--- | :--- | :--- |
| `attn_k.weight`, `attn_v.weight` | All `52/52` (`blk.0..51`) | **`Q8_0`** | 104 |
| `attn_q.weight`, `attn_output.weight` | All `52/52` (`blk.0..51`) | **`IQ4_XS` / `Q6_K`** | 104 |
| `ffn_down.weight` (residual stream writer) | All `52/52` (`blk.0..51`) | **`IQ4_XS` / `Q6_K`** | 52 |
| `ffn_gate.weight`, `ffn_up.weight` | `44/52` layers | **`IQ4_XS`** | 88 |
| `ffn_gate.weight`, `ffn_up.weight` | `8/52` (`blk.17..20, 31..34`) | **`IQ2_XXS`** | 16 |
| Norms & Embeddings (`token_embd`, `output`) | Global | `F32` / `IQ4_XS` / `Q6_K` | 107 |

---

## Optimal `llama-server` Command for RTX 3090 / 4090 (24 GB)

```bash
llama-server \
  --model Nanbeige4.3-33B-Hyper.gguf \
  --alias Nanbeige4.3-33B-Hyper \
  --host 127.0.0.1 --port 8091 \
  -ngl 99 -fit off \
  -c 57344 -np 4 --kv-unified --cont-batching \
  -ctk q8_0 -ctv q8_0 -fa on \
  --reasoning on --reasoning-format deepseek --no-reasoning-preserve \
  --temp 0.2 --top-p 0.90 --top-k 0 --min-p 0.05 \
  -b 2048 -ub 512 \
  -bs --spec-draft-backend-sampling \
  --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 12 \
  --cache-reuse 256 --slots --metrics
```

### Key Performance Notes
1. **Use symmetric `-ctk q8_0 -ctv q8_0` (NOT `-ctk q8_0 -ctv q4_0`):** Standard `ggml-cuda` builds compile native FlashAttention kernels for symmetric `q8_0/q8_0`. Asymmetric `q8_0/q4_0` triggers a slow `f16` fallback.
2. **Pass `--no-reasoning-preserve`:** Prevents `llama-server` from re-injecting prior turns' `<think>...</think>` traces into the prompt history across multi-step agent loops.
3. **Draftless N-Gram Speculative Decoding (`--spec-type ngram-simple`):** Uses 0 MB extra VRAM and boosts code/JSON generation to **45–87 tok/s** on a single RTX 3090.
