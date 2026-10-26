#!/usr/bin/env python3
"""
HyperNanbeige-33B Server Launcher & Multi-Stream Benchmark
==========================================================
Launches llama-server with symmetric Q8_0 KV cache (enabling native CUDA FlashAttention
vec_case<128, Q8_0, Q8_0>), 4-slot unified KV pool, N-gram speculative decoding with
GPU backend sampling, and DeepSeek reasoning format without history re-injection.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_vram() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, free, total = [int(x.strip()) for x in out.splitlines()[0].split(",")]
        return {"used_mib": used, "free_mib": free, "total_mib": total}
    except Exception:
        return {}


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.llama_server),
        "--model", str(args.model),
        "--alias", args.alias,
        "--host", args.host,
        "--port", str(args.port),
        "-ngl", "99",
        "-fit", "off",
        "-c", str(args.ctx_size),
        "-np", str(args.parallel),
        "--kv-unified",
        "--cont-batching",
        "-ctk", args.kv_type,
        "-ctv", args.kv_type,
        "-fa", "on",
        "-t", str(args.threads),
        "-tb", str(args.threads_batch),
        "--threads-http", "6",
        "--reasoning", "on",
        "--reasoning-format", args.reasoning_format,
        "--no-reasoning-preserve",
        "--temp", str(args.temp),
        "--top-p", str(args.top_p),
        "--top-k", "0",
        "--min-p", str(args.min_p),
        "-b", "2048",
        "-ub", "512",
        "-bs",
        "--spec-draft-backend-sampling",
        "--spec-type", "ngram-simple",
        "--spec-ngram-simple-size-n", "4",
        "--spec-ngram-simple-size-m", "12",
        "--cache-reuse", "256",
        "--slots",
        "--metrics",
    ]
    if args.reasoning_budget is not None:
        cmd.extend(["--reasoning-budget", str(args.reasoning_budget)])
    return cmd


def single_req(host: str, port: int, alias: str, idx: int, prompt: str, max_tokens: int = 256) -> dict:
    url = f"http://{host}:{port}/v1/chat/completions"
    body = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    dt = time.time() - t0
    timings = data.get("timings", {})
    usage = data.get("usage", {})
    pred_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
    pred_tps = timings.get("predicted_per_second", pred_n / max(dt, 0.001))
    return {
        "idx": idx,
        "elapsed_s": round(dt, 2),
        "tokens": pred_n,
        "slot_tps": round(pred_tps, 2),
    }


def run_benchmark(host: str, port: int, alias: str) -> None:
    prompts = [
        "Write a Python function `merge_k_sorted_lists` using `heapq` with type annotations and a brief docstring.",
        "Explain the difference between `ReplacingMergeTree` and `AggregatingMergeTree` in ClickHouse with an example.",
        "Solve for integers x, y: x^2 - y^2 = 45. List all positive integer solutions.",
        "Write a SQL query using `ROW_NUMBER()` and `LAG()` to find the day-over-day revenue change per store.",
    ]
    for n_parallel in (1, 2, 4):
        log(f"Running concurrency benchmark (n_parallel={n_parallel})...")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futs = [ex.submit(single_req, host, port, alias, i, prompts[i], 256) for i in range(n_parallel)]
            res = [f.result() for f in futs]
        wall_s = time.time() - t0
        total_tok = sum(r["tokens"] for r in res)
        agg_tps = round(total_tok / max(wall_s, 0.001), 2)
        avg_slot_tps = round(sum(r["slot_tps"] for r in res) / len(res), 2)
        log(
            f"  [n_parallel={n_parallel}] wall={wall_s:.2f}s | total_tokens={total_tok} | "
            f"aggregate={agg_tps} tok/s | avg_slot={avg_slot_tps} tok/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start HyperNanbeige-33B llama-server on a 24 GB GPU")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("HYPER_MODEL_PATH", "./models/Nanbeige4.3-33B-Hyper.gguf")),
        help="Path to Nanbeige4.3-33B-Hyper.gguf",
    )
    parser.add_argument(
        "--llama-server",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server")),
        help="Path to llama-server binary",
    )
    parser.add_argument("--alias", default="Nanbeige4.3-33B-Hyper")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--ctx-size", type=int, default=57344, help="Unified KV context pool size (default: 57344)")
    parser.add_argument("--parallel", type=int, default=4, help="Max concurrent slots (default: 4)")
    parser.add_argument("--kv-type", choices=["q8_0", "q4_0", "f16"], default="q8_0")
    parser.add_argument("--reasoning-format", choices=["deepseek", "none"], default="deepseek")
    parser.add_argument("--reasoning-budget", type=int, default=None, help="Optional max thinking tokens")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-batch", type=int, default=12)
    parser.add_argument("--log-file", type=pathlib.Path, default=pathlib.Path("llama-server-8091.log"))
    parser.add_argument("--benchmark", action="store_true", help="Run 1/2/4-stream throughput benchmark on startup")
    args = parser.parse_args()

    cmd = build_command(args)
    log(f"Starting: {' '.join(cmd)}")
    logf = open(args.log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"Spawned PID={proc.pid}, waiting for http://{args.host}:{args.port}/health ...")
    for _ in range(120):
        if proc.poll() is not None:
            tail = "\n".join(args.log_file.read_text(errors="replace").splitlines()[-25:])
            raise RuntimeError(f"llama-server exited early (rc={proc.returncode}):\n{tail}")
        try:
            with urllib.request.urlopen(f"http://{args.host}:{args.port}/health", timeout=2) as r:
                if r.status == 200:
                    log(f"Server is HEALTHY! VRAM: {get_vram()}")
                    break
        except Exception:
            time.sleep(1.0)
    else:
        raise RuntimeError("Timed out waiting for llama-server /health")

    if args.benchmark:
        run_benchmark(args.host, args.port, args.alias)
        log(f"VRAM after benchmark: {get_vram()}")


if __name__ == "__main__":
    main()
