"""The hipcc/nvcc NVFP4 GEMV must compute what the format says, not merely something.

The reference here is torch arithmetic over the same bytes -- dequantize, then matmul --
rather than the Triton kernel, for two reasons: it states the format's definition in one
readable place, and it runs on the devices this kernel exists for, where Triton cannot be
compiled at all and so could not serve as a reference even in principle.

"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _random_nvfp4(slots, n, k, device, gen):
    """Packed codes, e4m3 per-16 block scales and fp16 row globals for [slots, n, k]."""
    packed = torch.randint(0, 256, (slots, n, k // 2), dtype=torch.uint8,
                           device=device, generator=gen)
    # Scales as real e4m3 values, via the dtype rather than random bytes: a random byte
    # can be the NaN code, which is not something a checkpoint ever contains.
    scale = (torch.rand((slots, n, k // 16), device=device, generator=gen) * 3.0 + 0.25)
    scale = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    global_ = (torch.rand((slots, n), device=device, generator=gen) * 0.5 + 0.5).to(torch.float16)
    return packed, scale, global_


def _dequantize(packed, scale, global_, k):
    """The format's own definition: code -> E2M1, times its block scale, times the row global."""
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[((packed >> 4) & 0xF).long()]
    codes = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], k)
    s = scale.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=-1)
    return codes * s * global_.to(torch.float32).unsqueeze(-1)


def _reference(a, packed, scale, global_, topk_ids, top_k, k):
    w = _dequantize(packed, scale, global_, k)
    routes = topk_ids.reshape(-1)
    rows = [a[r // top_k].float() @ w[int(routes[r])].T for r in range(routes.numel())]
    return torch.stack(rows)


def test_matches_the_format_definition():
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260908)
    slots, n, k, tokens, top_k = 6, 64, 256, 3, 2

    a = torch.randn((tokens, k), device=device, dtype=torch.bfloat16, generator=gen)
    packed, scale, global_ = _random_nvfp4(slots, n, k, device, gen)
    topk_ids = torch.randint(0, slots, (tokens, top_k), dtype=torch.int32,
                             device=device, generator=gen)

    got = nvfp4_moe_vec(a, packed, scale, global_, topk_ids, top_k, n, tokens)
    want = _reference(a, packed, scale, global_, topk_ids, top_k, k)

    assert got.shape == (tokens * top_k, n)
    # bf16 activations and an fp32 accumulator: the tolerance is the input dtype's, not
    # the kernel's -- the weights are exact in both paths.
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_single_route_per_row_addresses_its_own_expert():
    """The down projection calls with top_k=1 and one activation row per route; the flat
    ids must still line up with the rows rather than being re-divided."""
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    slots, n, k, routes = 5, 32, 64, 6

    a = torch.randn((routes, k), device=device, dtype=torch.bfloat16, generator=gen)
    packed, scale, global_ = _random_nvfp4(slots, n, k, device, gen)
    ids = torch.randint(0, slots, (routes, 1), dtype=torch.int32, device=device, generator=gen)

    got = nvfp4_moe_vec(a, packed, scale, global_, ids, 1, n, routes)
    want = _reference(a, packed, scale, global_, ids, 1, k)
    torch.testing.assert_close(got.float(), want, rtol=2e-2, atol=2e-2)


def test_negative_ids_produce_no_output_read():
    """A route another executor owns carries a negative id and a zero weight; the kernel
    must not dereference it (a negative slot would walk off the front of the bank)."""
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(11)
    slots, n, k = 4, 32, 64
    a = torch.randn((2, k), device=device, dtype=torch.bfloat16, generator=gen)
    packed, scale, global_ = _random_nvfp4(slots, n, k, device, gen)
    ids = torch.tensor([[-1], [2]], dtype=torch.int32, device=device)

    got = nvfp4_moe_vec(a, packed, scale, global_, ids, 1, n, 2)
    assert got.isfinite().all()  # no wild read, and nothing poisons the finite half
    want = _reference(a[1:], packed, scale, global_, ids[1:], 1, k)
    torch.testing.assert_close(got[1].float(), want[0], rtol=2e-2, atol=2e-2)
