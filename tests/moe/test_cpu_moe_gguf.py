"""The CPU MoE executor on GGUF experts (``weight_format`` gguf): llama.cpp's CPU row dot
products over host banks at a ggml type per layer and projection.

Checked against the GPU dequantize of the same bytes: one token, and several tokens that
share experts (the executor then reads each expert once for all its routes), with routes
left to another executor (-1), on two layers of different types.
"""

from __future__ import annotations

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the reference")

H, I, E, TOP_K = 2560, 640, 8, 4
# per layer: (gate_up type, down type), as UD GGUFs mix them
LAYER_TYPES = [(18, 20), (23, 8)]  # (IQ3_XXS, IQ4_NL), (IQ4_XS, Q8_0)
# (ggml type) -> (block elements, block bytes)
BLOCKS = {8: (32, 34), 18: (256, 98), 20: (32, 18), 23: (256, 136)}


def _rows(ggml_type: int, rows: int, k: int) -> torch.Tensor:
    """Valid blocks: a small fp16 scale, random codes (IQ4_XS: small sub-block scales)."""
    block, block_bytes = BLOCKS[ggml_type]
    nb = k // block
    b = torch.randint(0, 256, (rows, nb, block_bytes), dtype=torch.uint8)
    b[:, :, :2] = (torch.rand(rows, nb) * 0.02 + 0.001).to(torch.float16).view(torch.uint8).view(rows, nb, 2)
    if ggml_type == 23:
        b[:, :, 2:4] = 0
    return b.view(rows, nb * block_bytes)


def _executor():
    from freetoken.kernel import ggml_cpu
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if not ggml_cpu.available():
        pytest.skip("no x86-64/aarch64 host with a C compiler")
    try:
        from freetoken.kernel import _cpu_moe  # noqa: F401
    except ImportError:
        pytest.skip("the _cpu_moe extension is not built")
    torch.manual_seed(0)
    gate_up = [torch.stack([_rows(gu, 2 * I, H) for _ in range(E)]).pin_memory() for gu, _ in LAYER_TYPES]
    down = [torch.stack([_rows(dn, H, I) for _ in range(E)]).pin_memory() for _, dn in LAYER_TYPES]
    cache = types.SimpleNamespace(
        num_layers=len(LAYER_TYPES), num_experts=E, quant_format="gguf", gguf_layer_types=LAYER_TYPES,
        bank_sources={"gate_up": gate_up, "down": down},
    )
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=4, device=torch.device("cuda"))
    return ex, gate_up, down


def _reference(layer, gate_up, down, x, ids, w):
    from freetoken.kernel.gguf import ggml_dequantize

    gu_t, dn_t = LAYER_TYPES[layer]
    ref = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(ids[t, k])
            if e < 0:
                continue
            g = ggml_dequantize(gate_up[layer][e].cuda(), gu_t, 2 * I, H, torch.float32).cpu()
            d = ggml_dequantize(down[layer][e].cuda(), dn_t, H, I, torch.float32).cpu()
            h = g @ x[t].float().cpu()
            ref[t] += float(w[t, k]) * (d @ (torch.nn.functional.silu(h[:I]) * h[I:]))
    return ref


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("routes", ["one-token", "shared-experts"])
def test_gguf_cpu_moe_matches_dequantized(layer, routes):
    ex, gate_up, down = _executor()
    if routes == "one-token":
        ids = torch.tensor([[3, 0, 6, 1]], dtype=torch.int32)
    else:  # experts 1 and 2 shared across tokens, one route handed elsewhere
        ids = torch.tensor([[0, 1, 2, 3], [1, 2, 5, -1], [2, 6, 0, 7]], dtype=torch.int32)
    T = ids.shape[0]
    x = (torch.randn(T, H) * 0.5).to(torch.bfloat16).cuda()
    w = torch.rand(T, TOP_K).cuda()
    y = ex.decode(layer, x, w, ids.cuda()).float().cpu()
    ref = _reference(layer, gate_up, down, x, ids, w)
    assert torch.isfinite(y).all()
    # the activations are quantized (Q8_K / Q8_0) and the intermediate is bf16, as in llama.cpp
    assert ((y - ref).norm() / ref.norm()).item() < 0.03
