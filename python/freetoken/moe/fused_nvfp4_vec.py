"""Grouped expert forward over native NVFP4 banks, built on the hipcc/nvcc GEMV.

The worker-side counterpart to :mod:`freetoken.moe.fused_nvfp4`, which orchestrates the
same arithmetic through Triton for the engine's own device. Separate because the two
share no code path: that one captures into a CUDA graph and dequantizes inside a Triton
K-loop, this one is a plain GEMV a worker process calls eagerly.

The same shape as :func:`fused_experts_gguf` -- gate_up GEMV, activation, down GEMV,
routed-weight combine -- so the two expert formats compose identically and a worker can
serve either by dispatching here on the format tag alone. What differs is only that an
NVFP4 layer is carried by three banks per projection (codes, block scales, row globals)
rather than one.
"""

from __future__ import annotations

import torch

from freetoken.moe.fused_q4_0 import _ACT


def fused_experts_nvfp4_vec(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,  # [num_slots, 2I, H//2]  uint8
    gate_up_scale: torch.Tensor,   # [num_slots, 2I, H//16] uint8 (fp8-e4m3)
    gate_up_global: torch.Tensor,  # [num_slots, 2I]        fp16
    down_packed: torch.Tensor,     # [num_slots, H, I//2]   uint8
    down_scale: torch.Tensor,      # [num_slots, H, I//16]  uint8 (fp8-e4m3)
    down_global: torch.Tensor,     # [num_slots, H]         fp16
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    act_fn=None,
    bufs: dict | None = None,
) -> torch.Tensor:
    """``act_fn`` overrides the activation implementation for callers that cannot use the
    compiled one -- a worker on a device Triton has no backend for. ``None`` keeps the
    default lookup, so nothing changes for anyone who does not ask.

    ``bufs`` makes the whole call allocation-free: every intermediate and the result are
    written into buffers the caller owns and reuses. A device computing beside a captured
    graph needs this -- while a capture is underway anywhere in the process its allocator
    may not ask the driver for memory, and it asks even for a block it already has cached.
    The keys are ``gate_up``, ``inter``, ``down``, ``s0``, ``s1``, ``wbf`` and ``out``; see
    :class:`~freetoken.moe.device_executor.DeviceMoeExecutor` for how they are sized.
    """
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec

    if bufs is None:
        if act_fn is None:
            act_fn = _ACT.get(activation)
        if act_fn is None:
            raise ValueError(f"unsupported MoE activation {activation!r}")

    # The block-scale banks carry their own fp8 type; the kernel reads them as bytes and
    # decodes the bits itself, which is the one path that works on every target (no
    # compilation target below sm_89 can type an fp8 pointer at all).
    if gate_up_scale.dtype != torch.uint8:
        gate_up_scale = gate_up_scale.view(torch.uint8)
    if down_scale.dtype != torch.uint8:
        down_scale = down_scale.view(torch.uint8)

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_packed.shape[1]  # 2 * intermediate
    h = down_packed.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    if bufs is not None:
        return _into_bufs(
            bufs, activation, hidden_states,
            gate_up_packed, gate_up_scale, gate_up_global,
            down_packed, down_scale, down_global,
            topk_weights, topk_ids, top_k, n2, h, num_tokens,
        )

    gate_up = nvfp4_moe_vec(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        topk_ids, top_k, n2, num_tokens,
    )
    inter = act_fn(gate_up)
    # Each of the num_tokens*top_k intermediate rows carries its own expert, so the down
    # projection is one route per row: top_k collapses to 1 and the flat ids still line up.
    out = nvfp4_moe_vec(
        inter, down_packed, down_scale, down_global,
        topk_ids, 1, h, num_tokens * top_k,
    )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def _into_bufs(
    bufs, activation, hidden_states,
    gate_up_packed, gate_up_scale, gate_up_global,
    down_packed, down_scale, down_global,
    topk_weights, topk_ids, top_k, n2, h, num_tokens,
):
    """The same four steps, every one of them writing into a buffer the caller owns."""
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec
    from freetoken.layers.activation_torch import INTO_BY_NAME

    act_into = INTO_BY_NAME.get(activation)
    if act_into is None:
        raise ValueError(
            f"MoE activation {activation!r} has no allocation-free form; a device that "
            f"computes beside a captured graph cannot serve this model"
        )
    routes = num_tokens * top_k
    gate_up = bufs["gate_up"][:routes]
    inter = bufs["inter"][:routes]
    down = bufs["down"][:routes]

    nvfp4_moe_vec(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        topk_ids, top_k, n2, num_tokens, out=gate_up,
    )
    act_into(gate_up, inter, bufs["s0"][:routes], bufs["s1"][:routes])
    nvfp4_moe_vec(
        inter, down_packed, down_scale, down_global,
        topk_ids, 1, h, routes, out=down,
    )
    # A route another executor owns is skipped by the kernel, so its rows keep whatever the
    # buffer held before. A zero routing weight does not make that harmless: NaN * 0 is NaN,
    # and bf16 memory that was never written is full of NaN encodings. Clear those rows.
    if "valid" in bufs:
        valid, invalid = bufs["valid"][:routes], bufs["invalid"][:routes]
        torch.ge(topk_ids.reshape(routes, 1), 0, out=valid)
        torch.logical_not(valid, out=invalid)
        down.masked_fill_(invalid, 0)

    # The routed weight, then the top_k sum -- both in place, both into buffers that are
    # already the right shape for this batch.
    wbf = bufs["wbf"][:routes]
    wbf.copy_(topk_weights.reshape(routes, 1))
    down.mul_(wbf)
    out = bufs["out"][:num_tokens]
    torch.sum(down.reshape(num_tokens, top_k, h), dim=1, out=out)
    return out


__all__ = ["fused_experts_nvfp4_vec"]
