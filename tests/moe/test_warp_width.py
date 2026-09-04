"""The GGUF grouped GEMV against a reference, on whatever wavefront width this machine has.

The MMVQ kernels reduce across a wavefront, so their launch width and their reduction have
to agree with the hardware. They did not: WARP_SIZE was 32 for everyone, which is right on
RDNA and NVIDIA and wrong on GCN and CDNA, where a wave is 64 lanes. The symptom was not a
wrong number but a hardware exception, because a shuffle then reaches lanes the launch never
started.

What makes this easy to get wrong twice is that WARP_SIZE_GGUF -- the quant block width, 32
quants per GGUF block -- is a different 32 that happens to look the same. It describes the
data and must not follow the hardware; the MMQ kernels tile by it and never shuffle, which
is why they stay 32 wide everywhere.

This test does not care which width it runs on: it checks the kernel against a dequantized
reference, so a launch that disagrees with the hardware fails wherever it is run.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.gguf.dequant import GGML_Q4_0


def _q4_0_bank(*shape: int, blocks: int) -> torch.Tensor:
    """Random but VALID Q4_0 bytes: 18 per block, the first two the fp16 scale.

    Random bytes in the scale decode to NaN often enough to make a comparison meaningless.
    """
    g = torch.Generator().manual_seed(1234)
    buf = torch.randint(0, 256, (*shape, blocks, 18), dtype=torch.uint8, generator=g)
    buf[..., :2] = torch.tensor([0.05], dtype=torch.float16).view(torch.uint8)
    return buf.reshape(*shape, blocks * 18)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_grouped_gemv_matches_dequantized_reference():
    from freetoken.kernel.gguf import ggml_dequantize, ggml_moe_a8_vec

    device = torch.device("cuda")
    E, H, N, tokens, top_k = 4, 128, 256, 3, 2  # N = the kernel's output width

    weights = _q4_0_bank(E, N, blocks=H // 32).to(device)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(tokens, H, generator=g).to(torch.bfloat16).to(device)
    ids = torch.randint(0, E, (tokens, top_k), generator=g, dtype=torch.int32).to(device)

    got = ggml_moe_a8_vec(x, weights, ids, top_k, int(GGML_Q4_0), N, tokens)

    # Reference: dequantize each routed expert and do the matmul in float.
    ref = torch.empty(tokens * top_k, N, dtype=torch.float32, device=device)
    for t in range(tokens):
        for j in range(top_k):
            e = int(ids[t, j])
            w = ggml_dequantize(weights[e], int(GGML_Q4_0), N, H, torch.float32)
            ref[t * top_k + j] = x[t].float() @ w.T

    # Not elementwise-equal, and not meant to be: the kernel quantizes the activations to
    # 8 bits on the way in (the "a8" in its name), which the float reference does not.
    # Measured agreement on a correct kernel is r = 0.99993 with a median relative error of
    # 1.4%. A wavefront disagreement does not land near this -- it reduces across lanes the
    # launch never started, which faults outright, and would destroy the correlation if it
    # somehow returned.
    got_f = got.float()
    assert torch.isfinite(got_f).all(), "kernel returned non-finite values"
    corr = torch.corrcoef(torch.stack([got_f.flatten(), ref.flatten()]))[0, 1]
    assert corr > 0.999, f"kernel disagrees with the dequantized reference (r={corr:.5f})"
    rel = ((got_f - ref).abs() / ref.abs().clamp(min=1e-3)).median()
    assert rel < 0.05, f"median relative error {rel:.4f} exceeds 8-bit activation noise"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_every_visible_device_agrees_on_the_result():
    """Devices of different wavefront widths must produce the same numbers.

    Skips unless this machine actually has two, which is the only place the bug shows: a
    single-width box cannot tell a hardcoded width from a correct one.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    widths = {
        i: torch.cuda.get_device_properties(i).warp_size
        for i in range(torch.cuda.device_count())
    }
    if len(set(widths.values())) < 2:
        pytest.skip(f"one wavefront width visible ({sorted(set(widths.values()))})")

    E, H, N, tokens, top_k = 4, 128, 256, 3, 2
    weights = _q4_0_bank(E, N, blocks=H // 32)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, E, (tokens, top_k), generator=g, dtype=torch.int32)

    results = {}
    for i in widths:
        dev = torch.device("cuda", i)
        out = ggml_moe_a8_vec(
            x.to(dev), weights.to(dev), ids.to(dev), top_k, int(GGML_Q4_0), N, tokens
        )
        results[i] = out.cpu()

    first = min(results)
    for i, out in results.items():
        torch.testing.assert_close(
            out, results[first], rtol=0, atol=0,
            msg=f"device {i} (wave{widths[i]}) disagrees with device {first} "
                f"(wave{widths[first]})",
        )
