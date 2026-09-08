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
) -> torch.Tensor:
    """``act_fn`` overrides the activation implementation for callers that cannot use the
    compiled one -- a worker on a device Triton has no backend for. ``None`` keeps the
    default lookup, so nothing changes for anyone who does not ask."""
    from freetoken.kernel.nvfp4_moe import nvfp4_moe_vec

    if act_fn is None:
        act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_packed.shape[1]  # 2 * intermediate
    h = down_packed.shape[1]  # hidden
    top_k = topk_ids.shape[1]

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


__all__ = ["fused_experts_nvfp4_vec"]
