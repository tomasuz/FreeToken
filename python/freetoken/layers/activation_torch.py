"""Gated activations in plain torch, for devices the compiled kernels cannot serve.

The fast path for these is a Triton kernel, and Triton does not target every architecture a
torch build does: a compiler that has no backend for a device fails at compile time, before
anything runs. That is a fine reason to lose a fused kernel and a poor reason to lose the
device, so these are the same functions expressed in ops every backend has.

They are **not** bit-identical to the kernels, and cannot be: the kernels use hardware
approximations (``ex2.approx``, ``tanh.approx``) where torch uses the exact functions. They
match to within those approximations -- close enough that a model's output is
indistinguishable, not close enough for an equality assertion.

The layout is the kernels': ``x`` is ``[..., 2d]`` with the gate in the first half and the
value in the second, and the result is ``[..., d]``. Arithmetic goes through float32
regardless of the input dtype, which is what the kernels do internally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1] // 2
    return x[..., :d].float(), x[..., d:].float()


def _finish(y: torch.Tensor, x: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    y = y.to(x.dtype)
    if out is None:
        return y
    out.copy_(y)
    return out


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    gate, up = _halves(x)
    return _finish(F.silu(gate) * up, x, out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Exact (erf) GELU gate, matching the kernels' default GELU branch."""
    gate, up = _halves(x)
    return _finish(F.gelu(gate) * up, x, out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    gate, up = _halves(x)
    return _finish(F.gelu(gate, approximate="tanh") * up, x, out)


def swigluoai_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    gate, up = _halves(x)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return _finish(gate * torch.sigmoid(alpha * gate) * (up + 1.0), x, out)


def swiglu_clamp_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    """``swigluoai`` without the ``(up + 1)`` bias."""
    gate, up = _halves(x)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return _finish(gate * torch.sigmoid(alpha * gate) * up, x, out)


# --- the same activations, without allocating -----------------------------------------
#
# A device that computes beside a captured graph may not allocate: a capture underway
# anywhere in the process stops that device's allocator from asking the driver for memory,
# and it asks even when a block of the right size is already cached. The functions above
# allocate several times each (``.float()``, the gate, the product), so the executor that
# runs beside a capture uses these instead: same arithmetic, same float32 interior, every
# result written into a buffer the caller already owns.
#
# ``s0`` and ``s1`` are float32 scratch of the output's shape, held by the caller for the
# life of the executor. They are read and rewritten freely; nothing survives the call.


def _gate_into(x, s0, s1):
    """``s0`` <- the gate half in float32, ``s1`` free. Shared prologue."""
    d = x.shape[-1] // 2
    s0.copy_(x[..., :d])
    return d


def _mul_up_into(x, d, s0, s1, out):
    """``out`` <- ``s0`` times the value half. Shared epilogue."""
    s1.copy_(x[..., d:])
    s0.mul_(s1)
    out.copy_(s0)
    return out


def silu_and_mul_into(x, out, s0, s1):
    d = _gate_into(x, s0, s1)
    torch.sigmoid(s0, out=s1)
    s0.mul_(s1)
    return _mul_up_into(x, d, s0, s1, out)


def gelu_and_mul_into(x, out, s0, s1):
    d = _gate_into(x, s0, s1)
    s1.copy_(s0)
    s1.mul_(0.7071067811865476)  # 1/sqrt(2)
    torch.erf(s1, out=s1)
    s1.add_(1.0)
    s0.mul_(s1)
    s0.mul_(0.5)
    return _mul_up_into(x, d, s0, s1, out)


def gelu_tanh_and_mul_into(x, out, s0, s1):
    d = _gate_into(x, s0, s1)
    s1.copy_(s0)
    s1.mul_(s1)
    s1.mul_(s0)
    s1.mul_(0.044715)
    s1.add_(s0)
    s1.mul_(0.7978845608028654)  # sqrt(2/pi)
    torch.tanh(s1, out=s1)
    s1.add_(1.0)
    s1.mul_(0.5)
    s0.mul_(s1)
    return _mul_up_into(x, d, s0, s1, out)


INTO_BY_NAME = {
    "silu": silu_and_mul_into,
    "gelu": gelu_and_mul_into,
    "gelu_tanh": gelu_tanh_and_mul_into,
    "gelu_pytorch_tanh": gelu_tanh_and_mul_into,
}


BY_NAME = {
    "silu": silu_and_mul,
    "gelu": gelu_and_mul,
    "gelu_tanh": gelu_tanh_and_mul,
    "gelu_pytorch_tanh": gelu_tanh_and_mul,
    "swigluoai": swigluoai_and_mul,
    "swiglu_clamp": swiglu_clamp_and_mul,
}


__all__ = [
    "BY_NAME",
    "INTO_BY_NAME",
    "silu_and_mul_into",
    "gelu_and_mul_into",
    "gelu_tanh_and_mul_into",
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "swigluoai_and_mul",
    "swiglu_clamp_and_mul",
]
