"""Gated activations in plain torch, against the compiled kernels.

These exist so a device the kernel compiler has no backend for stays usable. That is only
worth having if the numbers agree, so this checks them against the kernels wherever the
kernels run -- and checks the shape and layout contract everywhere, including on a machine
with no accelerator at all.

They are not bit-identical by construction: the kernels use hardware approximations
(``ex2.approx``, ``tanh.approx``) where torch uses the exact functions. The tolerance below
is what that difference costs, and it is far tighter than any real mistake would be.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.layers import activation_torch


def _x(rows: int = 7, d: int = 64, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    return (torch.randn(rows, 2 * d, generator=g) * 2).to(dtype)


@pytest.mark.parametrize(
    "name", ["silu", "gelu", "gelu_tanh", "swigluoai", "swiglu_clamp"]
)
def test_shape_and_layout_contract(name):
    """Output is the input's halved last dim, and the halves are gate then value."""
    fn = activation_torch.BY_NAME[name]
    x = _x()
    out = fn(x)
    assert out.shape == (7, 64)
    assert out.dtype == x.dtype

    # The value half scales the result linearly; the gate half does not. Zeroing the
    # value half must therefore zero the output, which pins down which half is which.
    x2 = x.clone()
    x2[:, 64:] = 0
    zeroed = fn(x2)
    expected_zero = name != "swigluoai"  # swigluoai adds 1 to the value half
    assert bool((zeroed == 0).all()) == expected_zero


def test_out_parameter_is_written_and_returned():
    x = _x()
    out = torch.empty(7, 64, dtype=x.dtype)
    got = activation_torch.silu_and_mul(x, out=out)
    assert got.data_ptr() == out.data_ptr(), "must write into the caller's buffer"
    torch.testing.assert_close(out, activation_torch.silu_and_mul(x))


def test_half_precision_computes_in_float32():
    """Accumulating in bf16 would lose more than the kernels do; the halves promote."""
    x = _x(dtype=torch.bfloat16)
    got = activation_torch.silu_and_mul(x)
    assert got.dtype == torch.bfloat16
    ref = activation_torch.silu_and_mul(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
@pytest.mark.parametrize("name", ["silu", "gelu", "gelu_tanh"])
def test_matches_the_compiled_kernel(name):
    from freetoken.moe.fused_q4_0 import _ACT

    kernel_fn = _ACT[name]
    x = _x(dtype=torch.bfloat16).cuda()
    try:
        want = kernel_fn(x)
    except Exception as exc:  # the very situation the fallback exists for
        pytest.skip(f"compiled {name} unavailable here: {type(exc).__name__}")
    got = activation_torch.BY_NAME[name](x)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
