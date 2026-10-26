#!/usr/bin/env python3
"""
Upload Nanbeige4.3-33B-Hyper.gguf & Model Card to Hugging Face Hub
==================================================================
Usage:
  export HF_TOKEN="hf_..."
  python3 upload_to_hf.py \
    --repo-id akhtyamovpavel/Nanbeige4.3-33B-Hyper-GGUF \
    --gguf /path/to/Nanbeige4.3-33B-Hyper.gguf \
    --card ./MODEL_CARD_HF.md
"""

import argparse
import os
import pathlib
import tempfile

# Avoid PermissionError if ~/.cache/huggingface/xet is owned by root
_cache_root = pathlib.Path(os.environ.get("HF_WRITE_CACHE", pathlib.Path(tempfile.gettempdir()) / f"hf_cache_{os.getuid()}"))
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_cache_root))
os.environ.setdefault("HF_HUB_CACHE", str(_cache_root / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(_cache_root / "xet"))
os.environ.setdefault("XET_CACHE_DIR", str(_cache_root / "xet"))

from huggingface_hub import HfApi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload HyperNanbeige-33B GGUF to Hugging Face")
    parser.add_argument("--repo-id", default="akhtyamovpavel/Nanbeige4.3-33B-Hyper-GGUF")
    parser.add_argument(
        "--gguf",
        type=pathlib.Path,
        default=pathlib.Path("./models/Nanbeige4.3-33B-Hyper.gguf"),
    )
    parser.add_argument(
        "--card",
        type=pathlib.Path,
        default=pathlib.Path("./MODEL_CARD_HF.md"),
    )
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Error: Set HF_TOKEN environment variable or pass --token hf_...")

    api = HfApi(token=args.token)
    print(f"[1/3] Creating/verifying Hugging Face model repo: {args.repo_id} ...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=False)

    if args.card.exists():
        print(f"[2/3] Uploading {args.card.name} -> README.md ...")
        api.upload_file(
            path_or_fileobj=str(args.card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update Model Card for Nanbeige4.3-33B-Hyper-GGUF (v2, 3.82 bpw)",
        )

    print(f"[3/3] Uploading {args.gguf} ({args.gguf.stat().st_size / (1024**3):.2f} GiB) ...")
    api.upload_file(
        path_or_fileobj=str(args.gguf),
        path_in_repo=args.gguf.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload Nanbeige4.3-33B-Hyper.gguf (v2, 16.71 GiB, 350/366 tensors IQ4_XS/Q8_0)",
    )
    print(f"Done! https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
