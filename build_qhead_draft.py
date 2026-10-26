#!/usr/bin/env python3
"""Re-quantize ONLY output.weight in Nanbeige4.3-33B-Draft-2L.gguf to Q4_0 and Q2_K using libggml-base.so."""

import ctypes
import os
import struct
import time
from build_draft_gguf import (
    GGUF_DEFAULT_ALIGNMENT,
    align_offset,
    pack_tensor_info,
    parse_gguf_with_block_count_offset,
)

GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q6_K = 14


def requantize_output_weight(src_gguf: str, dst_gguf: str, target_type: int) -> None:
    t0 = time.time()
    lib = ctypes.CDLL("/usr/local/lib/ollama/libggml-base.so")

    deq_q6k = lib.dequantize_row_q6_K
    deq_q6k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    deq_q6k.restype = None

    if target_type == GGML_TYPE_Q4_0:
        quant_fn = lib.quantize_row_q4_0_ref
        block_elems = 32
        block_bytes = 18
        type_name = "Q4_0"
    elif target_type == GGML_TYPE_Q2_K:
        quant_fn = lib.quantize_row_q2_K_ref
        block_elems = 256
        block_bytes = 84
        type_name = "Q2_K"
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")

    quant_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quant_fn.restype = None

    meta = parse_gguf_with_block_count_offset(src_gguf)
    with open(src_gguf, "rb") as f_src:
        kv_header = f_src.read(meta["kv_end"])

        # Preserve exact tensor ordering from src_gguf
        sorted_tensors = sorted(meta["tensors"].values(), key=lambda x: x["rel_offset"])
        prepared = []
        for tinfo in sorted_tensors:
            name = tinfo["name"]
            f_src.seek(meta["data_start"] + tinfo["rel_offset"])
            raw = f_src.read(tinfo["byte_size"])
            if name == "output.weight":
                assert tinfo["type"] == GGML_TYPE_Q6_K, f"Expected Q6_K (14), got {tinfo['type']}"
                n_elems = tinfo["dims"][0] * tinfo["dims"][1]
                n_q6k_blocks = n_elems // 256
                assert len(raw) == n_q6k_blocks * 210

                out_len = (n_elems // block_elems) * block_bytes
                out_buf = (ctypes.c_uint8 * out_len)()

                # Process in chunks of 4096 rows (4096 * 7168 = 29,360,128 floats)
                chunk_elems = 4096 * 7168
                f32_buf = (ctypes.c_float * chunk_elems)()
                in_view = memoryview(raw)

                offset_elems = 0
                while offset_elems < n_elems:
                    cur_elems = min(chunk_elems, n_elems - offset_elems)
                    in_byte_off = (offset_elems // 256) * 210
                    in_byte_len = (cur_elems // 256) * 210
                    out_byte_off = (offset_elems // block_elems) * block_bytes

                    in_chunk = (ctypes.c_uint8 * in_byte_len).from_buffer_copy(
                        in_view[in_byte_off : in_byte_off + in_byte_len]
                    )
                    deq_q6k(ctypes.byref(in_chunk), ctypes.byref(f32_buf), ctypes.c_int64(cur_elems))
                    quant_fn(
                        ctypes.byref(f32_buf),
                        ctypes.byref(out_buf, out_byte_off),
                        ctypes.c_int64(cur_elems),
                    )
                    offset_elems += cur_elems

                raw = bytes(out_buf)
                print(
                    f"[{type_name}] Re-quantized output.weight: "
                    f"{tinfo['byte_size']/1024**2:.1f} MiB -> {len(raw)/1024**2:.1f} MiB in {time.time()-t0:.1f}s",
                    flush=True,
                )
                prepared.append((name, tinfo["dims"], target_type, raw))
            else:
                prepared.append((name, tinfo["dims"], tinfo["type"], raw))

    cur_rel_offset = 0
    tensor_headers = []
    padded_lengths = []
    for name, dims, ggml_type, raw_bytes in prepared:
        tensor_headers.append(pack_tensor_info(name, dims, ggml_type, cur_rel_offset))
        padded_len = align_offset(len(raw_bytes), GGUF_DEFAULT_ALIGNMENT)
        padded_lengths.append(padded_len)
        cur_rel_offset += padded_len

    with open(dst_gguf, "wb") as f_out:
        f_out.write(kv_header)
        for th in tensor_headers:
            f_out.write(th)
        header_end = f_out.tell()
        data_start = align_offset(header_end, GGUF_DEFAULT_ALIGNMENT)
        if data_start > header_end:
            f_out.write(b"\x00" * (data_start - header_end))
        for (_, _, _, raw_bytes), padded_len in zip(prepared, padded_lengths):
            f_out.write(raw_bytes)
            pad = padded_len - len(raw_bytes)
            if pad > 0:
                f_out.write(b"\x00" * pad)

    print(f"Saved {dst_gguf} ({os.path.getsize(dst_gguf)/1024**3:.3f} GiB)", flush=True)


if __name__ == "__main__":
    base = "/home/akhtyamovpavel/nanbeige-33b-exp/models/Nanbeige4.3-33B-Draft-2L.gguf"
    requantize_output_weight(
        base,
        "/home/akhtyamovpavel/nanbeige-33b-exp/models/Nanbeige4.3-33B-Draft-2L-HeadQ4_0.gguf",
        GGML_TYPE_Q4_0,
    )
    requantize_output_weight(
        base,
        "/home/akhtyamovpavel/nanbeige-33b-exp/models/Nanbeige4.3-33B-Draft-2L-HeadQ2_K.gguf",
        GGML_TYPE_Q2_K,
    )
