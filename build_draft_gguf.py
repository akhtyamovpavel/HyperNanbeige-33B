#!/usr/bin/env python3
"""Build a 4-Layer Speculative Decoding Draft GGUF for Nanbeige4.3-33B.

Supports two modes:
  1. Distilled Safetensors -> GGUF (`--safetensors draft_layers_only.safetensors`):
     Takes the 4 distilled transformer layers trained by `train_draft_nanbeige.py`,
     applies HuggingFace-to-GGML RoPE head permutation on Q/K projections,
     quantizes attention matrices to Q8_0 and MLP matrices to Q4_0, and stitches
     them with the exact vocabulary embeddings and LM head from the base GGUF.
  2. Zero-Shot Layer Slicing (`--zero-shot-slice`):
     Slices anchor layers [0, 17, 34, 51] directly from an existing GGUF
     to verify structural compatibility with `llama-server --model-draft`.
"""

import argparse
import os
import pickle
import struct
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

GGUF_TYPE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def read_gguf_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def skip_gguf_value(f, vtype: int) -> None:
    if vtype in GGUF_TYPE_SIZES:
        f.seek(GGUF_TYPE_SIZES[vtype], os.SEEK_CUR)
    elif vtype == 8:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(length, os.SEEK_CUR)
    elif vtype == 9:
        atype, alen = struct.unpack("<IQ", f.read(12))
        if atype in GGUF_TYPE_SIZES:
            f.seek(GGUF_TYPE_SIZES[atype] * alen, os.SEEK_CUR)
        else:
            for _ in range(alen):
                skip_gguf_value(f, atype)
    else:
        raise ValueError(f"Unsupported GGUF metadata type: {vtype}")


def parse_gguf_with_block_count_offset(path: str) -> dict:
    """Parse GGUF header and locate the exact byte offset of `llama.block_count`."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a valid GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        block_count_val_offset = None
        for _ in range(n_kv):
            key = read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "llama.block_count":
                block_count_val_offset = f.tell()
            skip_gguf_value(f, vtype)

        kv_end = f.tell()
        tensors = {}
        for _ in range(n_tensors):
            name = read_gguf_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type, rel_offset = struct.unpack("<IQ", f.read(12))
            tensors[name] = {
                "name": name,
                "dims": dims,
                "type": ggml_type,
                "rel_offset": rel_offset,
            }

        data_start = align_offset(f.tell(), GGUF_DEFAULT_ALIGNMENT)
        file_size = os.path.getsize(path)

    sorted_t = sorted(tensors.values(), key=lambda x: x["rel_offset"])
    for idx, t in enumerate(sorted_t):
        if idx + 1 < len(sorted_t):
            t["byte_size"] = sorted_t[idx + 1]["rel_offset"] - t["rel_offset"]
        else:
            t["byte_size"] = file_size - (data_start + t["rel_offset"])

    return {
        "path": path,
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv_end": kv_end,
        "block_count_val_offset": block_count_val_offset,
        "data_start": data_start,
        "tensors": tensors,
    }


def permute_qk_hf_to_ggml(weight: np.ndarray, n_head: int) -> np.ndarray:
    """Convert HuggingFace Llama split-half RoPE Q/K weight layout to GGML interleaved layout."""
    out_features, in_features = weight.shape
    head_dim = out_features // n_head
    return (
        weight.reshape(n_head, 2, head_dim // 2, in_features)
        .SwapAxes(1, 2)
        if hasattr(weight, "SwapAxes")
        else weight.reshape(n_head, 2, head_dim // 2, in_features)
        .transpose(0, 2, 1, 3)
        .reshape(out_features, in_features)
    )


def quantize_q8_0(weight_f32: np.ndarray) -> bytes:
    """Vectorized GGML Q8_0 quantization (34 bytes per 32 floats: fp16 scale + 32 int8)."""
    flat = np.ascontiguousarray(weight_f32, dtype=np.float32).reshape(-1, 32)
    n_blocks = flat.shape[0]
    amax = np.max(np.abs(flat), axis=1)
    d_fp16 = (amax / 127.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    inv_d = np.where(d_f32 > 0.0, 1.0 / d_f32, 0.0)[:, None]
    qs = np.clip(np.rint(flat * inv_d), -127, 127).astype(np.int8)

    out = np.empty(n_blocks, dtype=[("d", "<f2"), ("qs", "i1", (32,))])
    out["d"] = d_fp16
    out["qs"] = qs
    return out.tobytes()


def quantize_q4_0(weight_f32: np.ndarray) -> bytes:
    """Vectorized GGML Q4_0 quantization (18 bytes per 32 floats: fp16 scale + 16 packed uint8)."""
    flat = np.ascontiguousarray(weight_f32, dtype=np.float32).reshape(-1, 32)
    n_blocks = flat.shape[0]
    amax = np.max(np.abs(flat), axis=1)
    d_fp16 = (amax / 7.0).astype(np.float16)
    d_f32 = d_fp16.astype(np.float32)
    inv_d = np.where(d_f32 > 0.0, 1.0 / d_f32, 0.0)[:, None]
    q = np.clip(np.rint(flat * inv_d) + 8.0, 0, 15).astype(np.uint8)
    packed = (q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)

    out = np.empty(n_blocks, dtype=[("d", "<f2"), ("qs", "u1", (16,))])
    out["d"] = d_fp16
    out["qs"] = packed
    return out.tobytes()


HF_TO_GGUF_LAYER_MAP = [
    ("input_layernorm.weight", "attn_norm.weight", "F32", None),
    ("self_attn.q_proj.weight", "attn_q.weight", "Q4_0", 56),
    ("self_attn.k_proj.weight", "attn_k.weight", "Q8_0", 8),
    ("self_attn.v_proj.weight", "attn_v.weight", "Q8_0", None),
    ("self_attn.o_proj.weight", "attn_output.weight", "Q4_0", None),
    ("post_attention_layernorm.weight", "ffn_norm.weight", "F32", None),
    ("mlp.gate_proj.weight", "ffn_gate.weight", "Q4_0", None),
    ("mlp.up_proj.weight", "ffn_up.weight", "Q4_0", None),
    ("mlp.down_proj.weight", "ffn_down.weight", "Q4_0", None),
]


def pack_tensor_info(name: str, dims: List[int], ggml_type: int, rel_offset: int) -> bytes:
    name_b = name.encode("utf-8")
    buf = [struct.pack("<Q", len(name_b)), name_b, struct.pack("<I", len(dims))]
    for d in dims:
        buf.append(struct.pack("<Q", d))
    buf.append(struct.pack("<IQ", ggml_type, rel_offset))
    return b"".join(buf)


def quantize_safetensors_layers(
    safetensors_path: str,
    num_draft_layers: int = 4,
) -> List[Tuple[str, List[int], int, bytes]]:
    import torch
    from safetensors.torch import load_file

    print(f"Loading distilled layers from {safetensors_path}...")
    st_dict = load_file(safetensors_path, device="cpu")
    quantized_layers: List[Tuple[str, List[int], int, bytes]] = []
    for dst_idx in range(num_draft_layers):
        print(f"Quantizing distilled layer {dst_idx}/{num_draft_layers - 1}...")
        for hf_suffix, gguf_suffix, qmode, n_head_perm in HF_TO_GGUF_LAYER_MAP:
            st_key = f"layers.{dst_idx}.{hf_suffix}"
            if st_key not in st_dict:
                st_key = f"model.layers.{dst_idx}.{hf_suffix}"
            w_np = st_dict[st_key].to(torch.float32).numpy()
            if n_head_perm is not None:
                w_np = permute_qk_hf_to_ggml(w_np, n_head_perm)

            if w_np.ndim == 1:
                dims = [w_np.shape[0]]
            else:
                out_f, in_f = w_np.shape
                dims = [in_f, out_f]

            if qmode == "F32":
                ggml_type = GGML_TYPE_F32
                raw_bytes = np.ascontiguousarray(w_np, dtype=np.float32).tobytes()
            elif qmode == "Q8_0":
                ggml_type = GGML_TYPE_Q8_0
                raw_bytes = quantize_q8_0(w_np)
            elif qmode == "Q4_0":
                ggml_type = GGML_TYPE_Q4_0
                raw_bytes = quantize_q4_0(w_np)
            else:
                raise ValueError(f"Unknown qmode: {qmode}")

            dst_name = f"blk.{dst_idx}.{gguf_suffix}"
            quantized_layers.append((dst_name, dims, ggml_type, raw_bytes))
    return quantized_layers


def build_draft_gguf(
    base_gguf_path: str,
    out_gguf_path: str,
    safetensors_path: str = None,
    import_quantized_path: str = None,
    anchor_layers: List[int] = [0, 17, 34, 51],
) -> None:
    meta = parse_gguf_with_block_count_offset(base_gguf_path)
    if meta["block_count_val_offset"] is None:
        raise RuntimeError("Could not find `llama.block_count` in base GGUF header")

    num_draft_layers = len(anchor_layers)
    with open(base_gguf_path, "rb") as f_base:
        kv_header = bytearray(f_base.read(meta["kv_end"]))

    # Patch n_tensors (bytes 8..16) and llama.block_count
    total_tensors = 3 + num_draft_layers * len(HF_TO_GGUF_LAYER_MAP)
    struct.pack_into("<Q", kv_header, 8, total_tensors)
    struct.pack_into("<I", kv_header, meta["block_count_val_offset"], num_draft_layers)

    prepared_tensors: List[Tuple[str, List[int], int, bytes]] = []

    # 1. Copy token_embd.weight, output_norm.weight, output.weight from base_gguf
    base_shared_names = ["token_embd.weight", "output_norm.weight", "output.weight"]
    with open(base_gguf_path, "rb") as f_base:
        for tname in base_shared_names:
            tinfo = meta["tensors"][tname]
            f_base.seek(meta["data_start"] + tinfo["rel_offset"])
            raw_bytes = f_base.read(tinfo["byte_size"])
            prepared_tensors.append((tname, tinfo["dims"], tinfo["type"], raw_bytes))

        if safetensors_path is None and import_quantized_path is None:
            # Mode B: Zero-shot slice of anchor layers directly from base_gguf
            for dst_idx, src_idx in enumerate(anchor_layers):
                for _, gguf_suffix, _, _ in HF_TO_GGUF_LAYER_MAP:
                    src_name = f"blk.{src_idx}.{gguf_suffix}"
                    dst_name = f"blk.{dst_idx}.{gguf_suffix}"
                    tinfo = meta["tensors"][src_name]
                    f_base.seek(meta["data_start"] + tinfo["rel_offset"])
                    raw_bytes = f_base.read(tinfo["byte_size"])
                    prepared_tensors.append((dst_name, tinfo["dims"], tinfo["type"], raw_bytes))

    if import_quantized_path is not None:
        print(f"Importing pre-quantized layers from {import_quantized_path}...")
        with open(import_quantized_path, "rb") as fq:
            prepared_tensors.extend(pickle.load(fq))
    elif safetensors_path is not None:
        prepared_tensors.extend(quantize_safetensors_layers(safetensors_path, num_draft_layers))

    # Compute relative offsets and write GGUF
    cur_rel_offset = 0
    tensor_headers = []
    padded_lengths = []
    for name, dims, ggml_type, raw_bytes in prepared_tensors:
        tensor_headers.append(pack_tensor_info(name, dims, ggml_type, cur_rel_offset))
        padded_len = align_offset(len(raw_bytes), GGUF_DEFAULT_ALIGNMENT)
        padded_lengths.append(padded_len)
        cur_rel_offset += padded_len

    os.makedirs(os.path.dirname(os.path.abspath(out_gguf_path)), exist_ok=True)
    with open(out_gguf_path, "wb") as f_out:
        f_out.write(kv_header)
        for th in tensor_headers:
            f_out.write(th)
        header_end = f_out.tell()
        data_start = align_offset(header_end, GGUF_DEFAULT_ALIGNMENT)
        if data_start > header_end:
            f_out.write(b"\x00" * (data_start - header_end))

        for (_, _, _, raw_bytes), padded_len in zip(prepared_tensors, padded_lengths):
            f_out.write(raw_bytes)
            pad = padded_len - len(raw_bytes)
            if pad > 0:
                f_out.write(b"\x00" * pad)

    size_gib = os.path.getsize(out_gguf_path) / (1024 ** 3)
    print(f"Done! Wrote {out_gguf_path} ({size_gib:.3f} GiB, {total_tensors} tensors, {num_draft_layers} layers)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 4-Layer Draft GGUF for Nanbeige4.3-33B")
    parser.add_argument("--base-gguf", default=None, help="Path to base IQ2_XXS or Hyper GGUF")
    parser.add_argument("--safetensors", default=None, help="Path to distilled draft_layers_only.safetensors")
    parser.add_argument("--export-quantized", default=None, help="Export quantized 4-layer bundle (.pkl)")
    parser.add_argument("--import-quantized", default=None, help="Import quantized 4-layer bundle (.pkl)")
    parser.add_argument("--layers", default="0,17,34,51", help="Comma-separated anchor layer indices")
    parser.add_argument("--zero-shot-slice", action="store_true", help="Slice anchor layers directly from base GGUF")
    parser.add_argument("--out", default=None, help="Output Draft GGUF path")
    args = parser.parse_args()

    anchor_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    if args.export_quantized:
        if not args.safetensors:
            raise SystemExit("--export-quantized requires --safetensors")
        qlayers = quantize_safetensors_layers(args.safetensors, num_draft_layers=len(anchor_layers))
        with open(args.export_quantized, "wb") as fq:
            pickle.dump(qlayers, fq, protocol=5)
        sz_gib = os.path.getsize(args.export_quantized) / (1024 ** 3)
        print(f"Exported quantized {len(anchor_layers)}-layer bundle to {args.export_quantized} ({sz_gib:.3f} GiB)")
        raise SystemExit(0)

    if not args.base_gguf or not args.out:
        raise SystemExit("--base-gguf and --out are required when building GGUF")

    build_draft_gguf(
        base_gguf_path=args.base_gguf,
        out_gguf_path=args.out,
        safetensors_path=args.safetensors,
        import_quantized_path=args.import_quantized,
        anchor_layers=anchor_layers,
    )
