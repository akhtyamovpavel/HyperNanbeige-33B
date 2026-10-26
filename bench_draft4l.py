#!/usr/bin/env python3
"""Benchmark Nanbeige4.3-33B-Hyper.gguf + Nanbeige4.3-33B-Draft-4L.gguf on RTX 3090."""

import argparse
import json
import subprocess
import time
import urllib.request

from switch_multithread import (
    BACKENDS_DIR,
    MODEL_PATH,
    OLLAMA_DIR,
    WEBUI_DIR,
    WORKDIR,
    get_env,
    get_vram,
    single_req,
    stop_existing,
)


def run_test(draft_name: str = "Nanbeige4.3-33B-Draft-4L.gguf"):
    stop_existing()
    draft_path = WORKDIR / "models" / draft_name
    log_path = WORKDIR / f"server_{draft_name}.log"

    cmd = [
        str(OLLAMA_DIR / "llama-server"),
        "--model", str(MODEL_PATH),
        "--model-draft", str(draft_path),
        "--alias", "Nanbeige4.3-33B-Hyper",
        "--host", "127.0.0.1",
        "--port", "8091",
        "-ngl", "99",
        "-ngld", "99",
        "-otd", "token_embd.weight=CPU",
        "-fit", "off",
        "-c", "36864",
        "-np", "2",
        "--kv-unified",
        "--cont-batching",
        "-ctk", "q8_0",
        "-ctv", "q8_0",
        "-ctkd", "q8_0",
        "-ctvd", "q8_0",
        "-fa", "on",
        "-t", "8",
        "-tb", "12",
        "-td", "6",
        "--threads-http", "6",
        "--reasoning", "on",
        "--reasoning-format", "deepseek",
        "--no-reasoning-preserve",
        "--temp", "0.2",
        "--top-p", "0.90",
        "--top-k", "0",
        "--min-p", "0.05",
        "-b", "2048",
        "-ub", "512",
        "-bs",
        "--spec-draft-backend-sampling",
        "--spec-type", "draft-simple",
        "--spec-draft-n-max", "5",
        "--spec-draft-n-min", "1",
        "--spec-draft-p-min", "0.45",
        "--cache-reuse", "256",
        "--slots",
        "--metrics",
        "--path", str(WEBUI_DIR),
    ]

    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=get_env(),
        cwd=str(BACKENDS_DIR),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Started PID={proc.pid}, waiting for /health...", flush=True)
    for _ in range(90):
        if proc.poll() is not None:
            print("EXITED EARLY:\n", log_path.read_text(errors="replace")[-2000:], flush=True)
            return
        try:
            with urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2) as r:
                if r.status == 200:
                    print("HEALTHY! VRAM:", get_vram(), flush=True)
                    break
        except Exception:
            time.sleep(1.0)

    prompts = [
        "Write a Python function `merge_k_sorted_lists` using `heapq` with type annotations and unit tests.",
        "Write a SQL query using `ROW_NUMBER()` and `LAG()` to compute day-over-day revenue delta per store.",
    ]
    for idx, p in enumerate(prompts):
        res = single_req(idx, p, max_tokens=256)
        print(f"Prompt {idx}: {res}", flush=True)

    lines = log_path.read_text(errors="replace").splitlines()
    draft_lines = [ln for ln in lines if "draft" in ln.lower() or "eval" in ln.lower()][-20:]
    print("--- Log tail ---", flush=True)
    for ln in draft_lines:
        print(ln, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="Nanbeige4.3-33B-Draft-4L.gguf")
    args = p.parse_args()
    run_test(args.draft)
