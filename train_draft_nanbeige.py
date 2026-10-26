#!/usr/bin/env python3
"""
Online Layer-Pruned Drafter Distillation for Nanbeige4.3-33B on RTX PRO 6000 (96 GB)
====================================================================================
Initializes a 4-layer (or 6-layer) student from evenly spaced anchor layers of
Nanbeige/Nanbeige4.3-33B ([0, 17, 34, 51]), shares the frozen embed_tokens, norm,
and lm_head in GPU memory, and performs Online Knowledge Distillation (Multi-Layer
Hidden Alignment + Pre-Head RMSNorm Alignment + Chunked Top-128 Symmetric KL)
entirely in 96 GB VRAM with zero disk bloat for intermediate logits.
"""

import argparse
import copy
import json
import math
import os
import pathlib
import time

# Default all caches to /mnt/h3-builds (1.6 TB NVMe) before importing huggingface/torch
DEFAULT_WORKDIR = pathlib.Path("/mnt/h3-builds/context/nanbeige_draft")
if DEFAULT_WORKDIR.parent.exists():
    os.environ.setdefault("HF_HOME", str(DEFAULT_WORKDIR / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_WORKDIR / "hf_cache" / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_WORKDIR / "hf_cache" / "xet"))
    os.environ.setdefault("TMPDIR", str(DEFAULT_WORKDIR / "tmp"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SEED_PROMPTS = [
    "Write a FastAPI microservice in Python with Pydantic v2 validation for a habit tracker (CRUD, mark done, streak stats).",
    "Проанализируй ошибку `ImportError: cannot import name ServerProtocol from websockets.server` при запуске uvicorn и предложи решение через virtualenv.",
    "Write an optimized SQL query using window functions (`LAG`, `ROW_NUMBER`, `SUM OVER`) to compute 7-day rolling retention and revenue delta.",
    "Объясни разницу между `ReplacingMergeTree` и `AggregatingMergeTree` в ClickHouse и напиши DDL для витрины событий.",
    "Implement a thread-safe LRU cache with TTL expiration in Python 3.12 using `OrderedDict` and `threading.RLock`, plus unit tests.",
    "Напиши скрипт на Python для парсинга заголовка GGUF-файла через `mmap` и `struct` с выводом списка тензоров и их размеров.",
    "Debug a Kubernetes `CrashLoopBackOff` where readinessProbe fails on port 8000 due to slow model weight initialization in FastAPI lifespan.",
    "Реализуй алгоритм `Regularized Evolution` для аппаратно-зависимого поиска архитектуры нейросети (Hardware-Aware NAS) под лимит SRAM <= 10 КБ.",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SharedDraftStudent(nn.Module):
    """4-layer (or K-layer) student sharing frozen embed_tokens, norm, and lm_head with Teacher."""

    def __init__(self, teacher: nn.Module, layer_indices: list[int]):
        super().__init__()
        self.config = copy.deepcopy(teacher.config)
        self.config.num_hidden_layers = len(layer_indices)
        self.layer_indices = layer_indices

        # Deep-copy only the selected transformer layers as trainable parameters
        cloned_layers = []
        for new_idx, src_idx in enumerate(layer_indices):
            layer_copy = copy.deepcopy(teacher.model.layers[src_idx])
            if hasattr(layer_copy, "self_attn") and hasattr(layer_copy.self_attn, "layer_idx"):
                layer_copy.self_attn.layer_idx = new_idx
            cloned_layers.append(layer_copy)

        self.layers = nn.ModuleList(cloned_layers)
        for p in self.layers.parameters():
            p.requires_grad_(True)

        # Shallow-copy backbone WITH its own _modules dict so teacher.model.layers is untouched!
        backbone = copy.copy(teacher.model)
        backbone._modules = teacher.model._modules.copy()
        backbone.config = self.config
        backbone.layers = self.layers
        self.backbone = backbone
        self.lm_head = teacher.lm_head

    @property
    def embed_tokens(self):
        return self.backbone.embed_tokens

    @property
    def norm(self):
        return self.backbone.norm

    def forward_hidden(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = self.backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        # out.hidden_states has (embed_out, layer_0_out, ..., layer_{K-1}_out)
        inter_h = list(out.hidden_states[1:])
        return out.last_hidden_state, inter_h


@torch.no_grad()
def teacher_forward_targets(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    out = teacher.model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    # out.hidden_states[idx + 1] is the output of teacher.model.layers[idx]
    ordered_inter = [out.hidden_states[idx + 1].detach() for idx in layer_indices]
    h_norm_t = out.last_hidden_state.detach()
    return h_norm_t, ordered_inter


def compute_distill_loss(
    student: SharedDraftStudent,
    h_norm_s: torch.Tensor,
    inter_s: list[torch.Tensor],
    h_norm_t: torch.Tensor,
    inter_t: list[torch.Tensor],
    topk: int = 128,
    temp: float = 1.5,
) -> tuple[torch.Tensor, dict]:
    # 1. Intermediate residual stream alignment (SmoothL1 + Cosine)
    inter_loss = torch.tensor(0.0, device=h_norm_s.device, dtype=torch.float32)
    for hs, ht in zip(inter_s[:-1], inter_t[:-1]):
        l1 = F.smooth_l1_loss(hs.float(), ht.float())
        cos = 1.0 - F.cosine_similarity(hs.float(), ht.float(), dim=-1).mean()
        inter_loss = inter_loss + l1 + 0.5 * cos
    inter_loss = inter_loss / max(1, len(inter_s) - 1)

    # 2. Pre-head RMSNorm hidden alignment
    norm_l1 = F.smooth_l1_loss(h_norm_s.float(), h_norm_t.float())
    norm_cos = 1.0 - F.cosine_similarity(h_norm_s.float(), h_norm_t.float(), dim=-1).mean()
    norm_loss = 8.0 * norm_l1 + 2.0 * norm_cos

    # 3. Chunked Top-K Symmetric KL Divergence through shared lm_head
    seq_len = h_norm_s.shape[1]
    chunk_sz = 128
    kl_total = torch.tensor(0.0, device=h_norm_s.device, dtype=torch.float32)
    top1_hits = 0
    top5_hits = 0
    total_toks = 0
    n_chunks = 0

    for start in range(0, seq_len, chunk_sz):
        end = min(seq_len, start + chunk_sz)
        hs_c = h_norm_s[:, start:end, :]
        ht_c = h_norm_t[:, start:end, :]

        with torch.no_grad():
            logits_t = student.lm_head(ht_c)
            tk_vals_t, tk_idx = torch.topk(logits_t, k=topk, dim=-1)
            top1_t = tk_idx[:, :, 0]

        logits_s = student.lm_head(hs_c)
        with torch.no_grad():
            top5_s = torch.topk(logits_s, k=5, dim=-1).indices
            top1_s = top5_s[:, :, 0]
            top1_hits += (top1_s == top1_t).sum().item()
            top5_hits += (top5_s == top1_t.unsqueeze(-1)).any(dim=-1).sum().item()
            total_toks += top1_t.numel()

        tk_vals_s = torch.gather(logits_s, dim=-1, index=tk_idx).float()
        tk_vals_t_f = tk_vals_t.float()

        log_p_t = F.log_softmax(tk_vals_t_f / temp, dim=-1)
        log_p_s = F.log_softmax(tk_vals_s / temp, dim=-1)
        p_t = log_p_t.exp()
        p_s = log_p_s.exp()

        fwd_kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temp ** 2)
        rev_kl = F.kl_div(log_p_t, p_s, reduction="batchmean") * (temp ** 2)
        kl_total = kl_total + 0.5 * (fwd_kl + rev_kl)
        n_chunks += 1

    kl_loss = kl_total / max(1, n_chunks)
    total_loss = kl_loss + norm_loss + 0.5 * inter_loss
    metrics = {
        "loss": round(total_loss.item(), 4),
        "kl": round(kl_loss.item(), 4),
        "norm": round(norm_loss.item(), 4),
        "top1_acc": round(100.0 * top1_hits / max(1, total_toks), 2),
        "top5_acc": round(100.0 * top5_hits / max(1, total_toks), 2),
    }
    return total_loss, metrics


def build_token_batches(
    tokenizer,
    traces_jsonl: pathlib.Path,
    seq_len: int = 512,
    device: str = "cuda",
) -> list[torch.Tensor]:
    texts = []
    if traces_jsonl.exists():
        for line in traces_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                if obj.get("text"):
                    texts.append(obj["text"])
    for p in SEED_PROMPTS:
        texts.append(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n")

    all_ids = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True)
        all_ids.extend(ids)

    batches = []
    for i in range(0, len(all_ids) - seq_len, seq_len // 2):
        chunk = all_ids[i : i + seq_len]
        if len(chunk) == seq_len:
            batches.append(torch.tensor([chunk], dtype=torch.long, device=device))
    return batches


def save_draft_checkpoint(
    student: SharedDraftStudent,
    tokenizer,
    out_dir: pathlib.Path,
    save_full_hf: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    student.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    state_dict = {}
    for idx, layer in enumerate(student.layers):
        for k, v in layer.state_dict().items():
            state_dict[f"model.layers.{idx}.{k}"] = v.detach().cpu().contiguous()

    if save_full_hf:
        state_dict["model.embed_tokens.weight"] = student.embed_tokens.weight.detach().cpu().contiguous()
        state_dict["model.norm.weight"] = student.norm.weight.detach().cpu().contiguous()
        state_dict["lm_head.weight"] = student.lm_head.weight.detach().cpu().contiguous()
        save_file(state_dict, str(out_dir / "model.safetensors"))
        log(f"Saved full standalone HF checkpoint to {out_dir / 'model.safetensors'}")
    else:
        save_file(state_dict, str(out_dir / "draft_layers_only.safetensors"))
        log(f"Saved trained 4-layer weights to {out_dir / 'draft_layers_only.safetensors'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Online Distillation of 4-Layer Nanbeige4.3 Draft Model")
    parser.add_argument("--teacher", default="Nanbeige/Nanbeige4.3-33B")
    parser.add_argument(
        "--traces",
        type=pathlib.Path,
        default=pathlib.Path("./data/nanbeige_agentic_traces.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_WORKDIR / "Nanbeige4.3-Draft-4L",
    )
    parser.add_argument("--layers", default="0,17,34,51", help="Anchor layer indices from 52-layer teacher")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--save-full-hf", action="store_true", help="Also include embed_tokens and lm_head in safetensors")
    args = parser.parse_args()

    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    log(f"Loading tokenizer & teacher {args.teacher} in bfloat16 on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    log(f"Teacher loaded! VRAM allocated: {vram_gb:.2f} GiB | Initializing {len(layer_indices)}-layer student {layer_indices}...")
    student = SharedDraftStudent(teacher, layer_indices).to("cuda:0", dtype=torch.bfloat16)
    assert len(teacher.model.layers) != len(student.backbone.layers), "Teacher layers must remain untouched!"
    log(f"Verified: Teacher has {len(teacher.model.layers)} layers, Student has {len(student.backbone.layers)} layers.")

    batches = build_token_batches(tokenizer, args.traces, seq_len=args.seq_len, device="cuda:0")
    log(f"Prepared {len(batches)} overlapping training windows (seq_len={args.seq_len}, total tokens={len(batches)*args.seq_len:,})")

    optimizer = torch.optim.AdamW(student.layers.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(batches) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    step = 0
    for epoch in range(1, args.epochs + 1):
        ep_loss = 0.0
        ep_top1 = 0.0
        ep_top5 = 0.0
        t0 = time.time()
        for b_idx, input_ids in enumerate(batches, 1):
            step += 1
            optimizer.zero_grad(set_to_none=True)
            h_norm_t, inter_t = teacher_forward_targets(teacher, input_ids, layer_indices)
            h_norm_s, inter_s = student.forward_hidden(input_ids)
            loss, metrics = compute_distill_loss(student, h_norm_s, inter_s, h_norm_t, inter_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.layers.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += metrics["loss"]
            ep_top1 += metrics["top1_acc"]
            ep_top5 += metrics["top5_acc"]
            if b_idx % 20 == 0 or b_idx == len(batches):
                log(
                    f"Epoch {epoch}/{args.epochs} [{b_idx}/{len(batches)}] | "
                    f"loss={metrics['loss']:.4f} (kl={metrics['kl']:.4f}, norm={metrics['norm']:.4f}) | "
                    f"Top-1={metrics['top1_acc']:.1f}% | Top-5={metrics['top5_acc']:.1f}%"
                )
        dt = time.time() - t0
        n = max(1, len(batches))
        log(
            f"=== Epoch {epoch} Done in {dt:.1f}s | Avg Loss={ep_loss/n:.4f} | "
            f"Avg Top-1 Match={ep_top1/n:.2f}% | Avg Top-5 Match={ep_top5/n:.2f}% ==="
        )
        save_draft_checkpoint(student, tokenizer, args.out_dir, save_full_hf=args.save_full_hf)


if __name__ == "__main__":
    main()
