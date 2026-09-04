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
    "silu_and_mul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "swigluoai_and_mul",
    "swiglu_clamp_and_mul",
]
