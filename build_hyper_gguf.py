#!/usr/bin/env python3
"""
HyperNanbeige-33B GGUF Builder (Zero-Dependency Tensor Surgery)
===============================================================
Constructs a surgical mixed-precision GGUF for Nanbeige4.3-33B (52 layers, GQA, SwiGLU)
tailored to fit a single 24 GB GPU (RTX 3090 / RTX 4090) alongside a 57k–98k Q8_0 KV cache.

Profiles:
  - v2-quality (default, 16.71 GiB, ~3.82 bpw):
      Transplants 350 / 366 tensors from IQ4_XS (100% of attn_q/k/v/output, 100% of ffn_down,
      and 44/52 layers of ffn_gate/ffn_up), compressing only 8 non-anchor FFN gate/up blocks
      (layers 17..20 and 31..34) to IQ2_XXS. Eliminates sub-3-bit token jitter on Cyrillic/code
      while leaving ~6.6 GiB VRAM for a 57,344-token symmetric Q8_0 KV cache on 24 GB VRAM.
  - v1-longctx (10.94 GiB, ~2.28 bpw):
      Transplants 196 / 366 tensors from IQ4_XS (all attn_k, attn_v, attn_output across all
      52 layers + full attn_q/ffn in 14 anchor layers: 0..5, 24..27, 48..51). Leaves ~12 GiB
      VRAM for up to 98k–131k context.
"""

import argparse
import json
import mmap
import pathlib
import re
import struct
import time

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}

V2_COMPRESS_LAYERS = set(range(17, 21)) | set(range(31, 35))
V1_ANCHOR_LAYERS = set(range(0, 6)) | set(range(24, 28)) | set(range(48, 52))


def skip_gguf_value(mm: mmap.mmap, pos: int, vtype: int) -> int:
    if vtype in SCALAR_SIZES:
        return pos + SCALAR_SIZES[vtype]
    if vtype == 8:
        slen = struct.unpack_from("<Q", mm, pos)[0]
        return pos + 8 + slen
    if vtype == 9:
        etype, count = struct.unpack_from("<IQ", mm, pos)
        pos += 12
        if etype in SCALAR_SIZES:
            return pos + count * SCALAR_SIZES[etype]
        for _ in range(count):
            pos = skip_gguf_value(mm, pos, etype)
        return pos
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def parse_gguf_layout(path: pathlib.Path) -> dict:
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != b"GGUF":
            raise ValueError(f"Invalid GGUF magic in {path}")
        version, tensor_count, kv_count = struct.unpack_from("<IQQ", mm, 4)
        pos = 24
        alignment = 32
        for _ in range(kv_count):
            klen = struct.unpack_from("<Q", mm, pos)[0]
            pos += 8
            key = mm[pos : pos + klen].decode("utf-8", errors="replace")
            pos += klen
            vtype = struct.unpack_from("<I", mm, pos)[0]
            pos += 4
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack_from("<I", mm, pos)[0]
            pos = skip_gguf_value(mm, pos, vtype)
        kv_end = pos
        tensors = []
        for _ in range(tensor_count):
            nlen = struct.unpack_from("<Q", mm, pos)[0]
            pos += 8
            name = mm[pos : pos + nlen].decode("utf-8")
            pos += nlen
            n_dims = struct.unpack_from("<I", mm, pos)[0]
            pos += 4
            dims = struct.unpack_from(f"<{n_dims}Q", mm, pos)
            pos += 8 * n_dims
            ttype, offset = struct.unpack_from("<IQ", mm, pos)
            pos += 12
            tensors.append({
                "name": name,
                "n_dims": n_dims,
                "dims": dims,
                "type": ttype,
                "offset": offset,
            })
        data_start = (pos + alignment - 1) // alignment * alignment
        for i, t in enumerate(tensors):
            abs_start = data_start + t["offset"]
            abs_end = (data_start + tensors[i + 1]["offset"]) if (i + 1 < len(tensors)) else file_size
            t["abs_start"] = abs_start
            t["span"] = abs_end - abs_start
        mm.close()
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "alignment": alignment,
        "kv_end": kv_end,
        "data_start": data_start,
        "tensors": tensors,
    }


def should_upgrade_tensor(name: str, profile: str) -> bool:
    if name in ("token_embd.weight", "output_norm.weight", "output.weight"):
        return True
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not m:
        return True
    layer = int(m.group(1))
    suffix = m.group(2)

    if profile == "v2-quality":
        if suffix in ("ffn_gate.weight", "ffn_up.weight") and layer in V2_COMPRESS_LAYERS:
            return False
        return True

    if profile == "v1-longctx":
        if layer in V1_ANCHOR_LAYERS:
            return True
        if suffix in (
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ):
            return True
        return False

    raise ValueError(f"Unknown profile: {profile}")


def build_hyper_gguf(
    base_gguf: pathlib.Path,
    donor_gguf: pathlib.Path,
    output_gguf: pathlib.Path,
    profile: str = "v2-quality",
) -> dict:
    t0 = time.time()
    print(f"[1/4] Parsing base GGUF ({base_gguf.name})...")
    base_meta = parse_gguf_layout(base_gguf)
    print(f"[2/4] Parsing donor GGUF ({donor_gguf.name})...")
    donor_meta = parse_gguf_layout(donor_gguf)
    donor_map = {t["name"]: t for t in donor_meta["tensors"]}

    alignment = base_meta["alignment"]
    chosen_plan = []
    cur_data_offset = 0
    upgraded_count = 0
    type_hist = {}

    for bt in base_meta["tensors"]:
        name = bt["name"]
        dt = donor_map[name]
        use_donor = should_upgrade_tensor(name, profile)
        src_t = dt if use_donor else bt
        if use_donor:
            upgraded_count += 1
        tname = GGML_TYPE_NAMES.get(src_t["type"], f"TYPE_{src_t['type']}")
        type_hist[tname] = type_hist.get(tname, 0) + 1
        chosen_plan.append({
            "name": name,
            "n_dims": src_t["n_dims"],
            "dims": src_t["dims"],
            "type": src_t["type"],
            "new_offset": cur_data_offset,
            "src": "donor" if use_donor else "base",
            "abs_start": src_t["abs_start"],
            "span": src_t["span"],
        })
        cur_data_offset += src_t["span"]

    print(
        f"[3/4] Writing {output_gguf.name} "
        f"(profile={profile}, upgraded={upgraded_count}/{len(chosen_plan)} tensors)..."
    )
    with open(base_gguf, "rb") as fb, open(donor_gguf, "rb") as fd, open(output_gguf, "wb") as fout:
        mm_base = mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ)
        mm_donor = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)

        fout.write(mm_base[: base_meta["kv_end"]])
        for item in chosen_plan:
            nb = item["name"].encode("utf-8")
            fout.write(struct.pack("<Q", len(nb)))
            fout.write(nb)
            fout.write(struct.pack("<I", item["n_dims"]))
            fout.write(struct.pack(f"<{item['n_dims']}Q", *item["dims"]))
            fout.write(struct.pack("<IQ", item["type"], item["new_offset"]))

        pos = fout.tell()
        pad = (alignment - (pos % alignment)) % alignment
        if pad:
            fout.write(b"\x00" * pad)

        chunk_size = 64 * 1024 * 1024
        for item in chosen_plan:
            mm_src = mm_donor if item["src"] == "donor" else mm_base
            start = item["abs_start"]
            rem = item["span"]
            while rem > 0:
                step = min(rem, chunk_size)
                fout.write(mm_src[start : start + step])
                start += step
                rem -= step

        mm_base.close()
        mm_donor.close()

    total_bytes = output_gguf.stat().st_size
    elapsed = round(time.time() - t0, 2)
    summary = {
        "profile": profile,
        "output_file": str(output_gguf),
        "size_bytes": total_bytes,
        "size_gib": round(total_bytes / (1024 ** 3), 3),
        "tensors_total": len(chosen_plan),
        "tensors_from_iq4_xs": upgraded_count,
        "tensors_from_iq2_xxs": len(chosen_plan) - upgraded_count,
        "tensor_type_histogram": type_hist,
        "elapsed_sec": elapsed,
    }
    print(f"[4/4] Done in {elapsed}s -> {summary['size_gib']} GiB")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Build HyperNanbeige-33B mixed-precision GGUF")
    parser.add_argument("--base", required=True, type=pathlib.Path, help="Path to IQ2_XXS GGUF")
    parser.add_argument("--donor", required=True, type=pathlib.Path, help="Path to IQ4_XS GGUF")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output path for Hyper GGUF")
    parser.add_argument(
        "--profile",
        choices=["v2-quality", "v1-longctx"],
        default="v2-quality",
        help="Mixed-precision allocation profile (default: v2-quality)",
    )
    args = parser.parse_args()
    build_hyper_gguf(args.base, args.donor, args.output, args.profile)


if __name__ == "__main__":
    main()
