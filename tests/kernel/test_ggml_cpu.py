"""llama.cpp's CPU row dot products (kernel/ggml_cpu.py), as the CPU MoE executor calls them.

Each type's rows are built by hand -- a small block scale, random codes -- so every block is
valid, and the dot with a random activation row is checked against the GPU dequantize of the
same bytes. The activation is quantized (Q8_0 / Q8_K), as llama.cpp's CPU backend does.
"""

import ctypes

import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the reference")

# (ggml type, block elements, block bytes, offset of the fp16 scale)
_TYPES = {"q8_0": (8, 32, 34), "iq4_nl": (20, 32, 18), "iq3_xxs": (18, 256, 98), "iq4_xs": (23, 256, 136)}


def _rows(ggml_type: int, block: int, block_bytes: int, rows: int, k: int) -> torch.Tensor:
    nb = k // block
    b = torch.randint(0, 256, (rows, nb, block_bytes), dtype=torch.uint8)
    b[:, :, :2] = (torch.rand(rows, nb) * 0.02 + 0.001).to(torch.float16).view(torch.uint8).view(rows, nb, 2)
    if ggml_type == 23:  # iq4_xs: scales_h (2 bytes) then scales_l; keep the 6-bit scales small
        b[:, :, 2:4] = 0
    return b.view(rows, nb * block_bytes)


@GPU
@pytest.mark.parametrize("name", sorted(_TYPES))
def test_row_dot_matches_dequantized(name):
    from freetoken.kernel import ggml_cpu
    from freetoken.kernel.gguf import ggml_dequantize

    if not ggml_cpu.available():
        pytest.skip("no x86-64 C compiler")
    torch.manual_seed(0)
    ggml_type, block, block_bytes = _TYPES[name]
    rows, k = 64, 2560
    w = _rows(ggml_type, block, block_bytes, rows, k)
    x = torch.randn(k)
    rd = ggml_cpu.row_dot(ggml_type)
    assert rd is not None
    xq = torch.empty(rd.act_row_bytes(k), dtype=torch.uint8)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(rd.quantize)(
        x.data_ptr(), xq.data_ptr(), k)
    dot = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                           ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)(rd.vec_dot)
    out = torch.empty(rows)
    for r in range(rows):
        s = ctypes.c_float()
        dot(k, ctypes.addressof(s), 0, w[r].data_ptr(), 0, xq.data_ptr(), 0, 1)
        out[r] = s.value
    ref = ggml_dequantize(w.cuda(), ggml_type, rows, k, torch.float32).cpu() @ x
    assert torch.isfinite(out).all()
    assert ((out - ref).norm() / ref.norm()).item() < 0.02


# ggml types checked SIMD against llama.cpp's portable C: every type the lookup serves.
# Rows come from llama.cpp's reference quantizer; IQ3_XXS (whose quantizer needs its grids set
# up) from random codes under a small scale, which is a valid block too.
_SIMD_TYPES = {"q4_0": 2, "q5_0": 6, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12, "q5_k": 13,
               "q6_k": 14, "iq4_nl": 20, "iq4_xs": 23, "iq3_xxs": 18}


def _dot(rd, k, w_row, act) -> float:
    fn = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                          ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)(rd.vec_dot)
    s = ctypes.c_float()
    fn(k, ctypes.addressof(s), 0, w_row.data_ptr(), 0, act.data_ptr(), 0, 1)
    return s.value


@pytest.mark.parametrize("name", sorted(_SIMD_TYPES))
def test_simd_row_dot_matches_portable_c(name):
    """The SIMD dot product of this host (x86 AVX2/AVX-512, aarch64 NEON) against llama.cpp's
    portable C one, on the same quantized rows and activations; CPU only."""
    from freetoken.gguf_quant import row_bytes
    from freetoken.kernel import ggml_cpu

    if not ggml_cpu.available():
        pytest.skip("no x86-64/aarch64 host with a C compiler")
    ggml_type, rows, k = _SIMD_TYPES[name], 16, 2560
    torch.manual_seed(0)
    rb = row_bytes(k, ggml_type)
    w = torch.empty(rows, rb, dtype=torch.uint8)
    if ggml_type == 18:
        w = _rows(18, 256, 98, rows, k)
    else:
        for r in range(rows):
            src = torch.randn(k) * 0.05
            assert ggml_cpu.quantize_ref(ggml_type, src.data_ptr(), w[r].data_ptr(), k)
    simd, portable = ggml_cpu.row_dot(ggml_type), ggml_cpu.row_dot(ggml_type, generic=True)
    assert simd.act_type == portable.act_type
    x = torch.randn(k)
    act = torch.empty(portable.act_row_bytes(k), dtype=torch.uint8)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(portable.quantize)(
        x.data_ptr(), act.data_ptr(), k)
    a = torch.tensor([_dot(simd, k, w[r], act) for r in range(rows)])
    b = torch.tensor([_dot(portable, k, w[r], act) for r in range(rows)])
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert ((a - b).norm() / b.norm()).item() < 1e-3
