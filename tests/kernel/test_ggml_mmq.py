"""llama.cpp's grouped MMQ, as FreeToken calls it: experts in strided slot regions.

Checked against dequantize-then-matmul on the same bytes. Q8_0 and IQ4_NL blocks are built
by hand (a scale and codes), so every block is valid; both 32-element types with K=640 make
MMQ read past each expert's last row, which the zero bytes after the region must absorb.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _blocks(rows: int, k: int, ggml_type: int, device) -> torch.Tensor:
    from freetoken.gguf_quant import GGML_IQ4_NL, GGML_Q8_0

    nb = k // 32
    size = {GGML_Q8_0: 34, GGML_IQ4_NL: 18}[ggml_type]
    b = torch.empty(rows, nb, size, dtype=torch.uint8)
    b[:, :, :2] = (torch.rand(rows, nb) * 0.02 + 0.001).to(torch.float16).view(torch.uint8).view(rows, nb, 2)
    b[:, :, 2:] = torch.randint(0, 256, (rows, nb, size - 2), dtype=torch.uint8)
    return b.view(rows, nb * size).to(device)


@pytest.mark.parametrize("ggml_type", [8, 20])  # Q8_0, IQ4_NL
@pytest.mark.parametrize("shared_row", [True, False])  # gate/up (one row per token) or down
def test_mmq_moe_matches_dequantized_matmul(ggml_type, shared_row):
    from freetoken.kernel import ggml_mmq
    from freetoken.kernel.gguf import ggml_dequantize

    if not ggml_mmq.available():
        pytest.skip("no hipcc / not ROCm")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    experts, rows, k, slots, tokens, used = 16, 256, 640, 40, 37, 4
    w = [_blocks(rows, k, ggml_type, dev) for _ in range(experts)]
    row_bytes = w[0].shape[1]
    width = rows * row_bytes
    region = torch.zeros(slots * width + 4096, dtype=torch.uint8, device=dev)
    # distinct slots, the last one included: its expert's overread lands in the tail
    slot_of = torch.cat([torch.randperm(slots - 1)[: experts - 1], torch.tensor([slots - 1])]).to(dev)
    for e in range(experts):
        region[int(slot_of[e]) * width : int(slot_of[e] + 1) * width] = w[e].reshape(-1)
    view = region[: slots * width].view(slots, width).as_strided((slots, rows, row_bytes), (width, row_bytes, 1))

    expert_ids = torch.stack([torch.randperm(experts)[:used] for _ in range(tokens)]).to(dev)
    x = torch.randn(tokens, 1 if shared_row else used, k, device=dev)
    out = ggml_mmq.mmq_moe(view, ggml_type, k, x, slot_of[expert_ids].to(torch.int32), experts)

    ref = torch.empty_like(out)
    for e in range(experts):
        we = ggml_dequantize(w[e], ggml_type, rows, k, torch.float32)
        t, u = (expert_ids == e).nonzero(as_tuple=True)
        ref[t, u] = x[t, 0 if shared_row else u] @ we.T
    assert torch.isfinite(out).all()
    # MMQ quantizes the activations to q8_1, as llama.cpp does
    assert ((out - ref).norm() / ref.norm()).item() < 0.02
