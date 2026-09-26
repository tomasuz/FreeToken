"""Grouped expert GEMM over native GGUF Q4_0 banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed Q4_0 block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).
"""

from __future__ import annotations

import functools
import os

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}

# From this many tokens on, the experts run as llama.cpp's grouped MMQ (kernel/ggml_mmq)
# where it serves the types: 6-8x the per-row MMVQ at a 300-token prefill on the RX 9060 XT.
# Below it (decode, MTP verify) MMVQ stays. 0 turns MMQ off.
_MMQ_MIN_TOKENS = int(os.getenv("FREETOKEN_GGUF_MMQ_MIN_TOKENS", "8") or 0)
# llama.cpp's MMVQ takes at most this many tokens per launch (MMVQ_MAX_BATCH_SIZE)
_MMVQ_MAX_TOKENS = 8


def _mmq_usable(num_tokens: int, *types: int) -> bool:
    if not _MMQ_MIN_TOKENS or num_tokens < _MMQ_MIN_TOKENS or not torch.cuda.is_available():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False  # ctypes launches with host-side scratch sizing; never inside a graph
    from freetoken.kernel import ggml_mmq

    try:
        return all(ggml_mmq.supports(t) for t in types)
    except Exception:  # noqa: BLE001 -- no toolchain / build failure: MMVQ still serves
        return False


# llama.cpp's current MMVQ (kernel/ggml_mmq) for 2-8 tokens (MTP verify), per projection where
# it measured faster on the RX 9060 XT (x 10 of Qwen3.8's experts): every down type (IQ4_NL
# 50 vs 73 us at two tokens, Q8_0 70 vs 107) and gate/up fused with its activation except
# IQ3_XXS, where the older kernel stays ahead (144 vs 180 us). One token keeps the older
# kernels: the kernel-level gain there (~0.6 ms a step) is eaten by the f32/int32 conversions
# around the call, measured 53.1 vs 52.0 ms/token end to end.
_NEW_MMVQ = os.getenv("FREETOKEN_GGUF_NEW_MMVQ", "1").strip().lower() not in {"0", "false", "no", "off"}
_OLD_MMVQ_GATE_UP_TYPES = frozenset({18})  # IQ3_XXS


@functools.cache
def _new_mmvq_serves(ggml_type: int) -> bool:
    if not _NEW_MMVQ or not torch.cuda.is_available():
        return False
    from freetoken.kernel import ggml_mmq

    return ggml_mmq.supports(ggml_type)


def _weighted_sum(out: torch.Tensor, topk_weights: torch.Tensor, num_tokens: int, top_k: int) -> torch.Tensor:
    """Sum of the routes' outputs by their router weights, a zero-weight route dropped
    rather than multiplied: a caller that hands a route to another executor keeps its
    slot id pointing somewhere, and in a slot region shared by layers of different ggml
    types that somewhere can decode to inf -- which times zero is NaN."""
    w = topk_weights.reshape(num_tokens, top_k, 1).to(out.dtype)
    return torch.where(w != 0, out * w, out.new_zeros(())).sum(dim=1)


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    ggml_type: int,
    act_fn=None,
    *,
    down_type: int | None = None,
    num_experts: int | None = None,
) -> torch.Tensor:
    """``act_fn`` overrides the activation implementation for callers that cannot use the
    compiled one -- a worker process on a device Triton has no backend for, say. ``None``
    keeps the default lookup, so nothing changes for anyone who does not ask.

    ``down_type`` is the down projection's ggml type when it differs from gate/up's
    (unsloth's UD quants: IQ3_XXS gate/up over an IQ4_NL or Q8_0 down); ``None`` means the
    same. The banks may be strided views -- a slot cache sized for the widest layer.
    ``num_experts`` (the distinct experts the routing can reach; defaults to the slot count)
    only tunes the grouped MMQ's tile size."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if act_fn is None:
        act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(ggml_type)
    qt_down = qt if down_type is None else int(down_type)

    if _mmq_usable(num_tokens, qt, qt_down):
        from freetoken.kernel.ggml_mmq import mmq_moe

        k_in = hidden_states.shape[1]
        ids = topk_ids.to(torch.int32)
        n_exp = num_experts or gate_up_q.shape[0]
        gate_up = mmq_moe(gate_up_q, qt, k_in, hidden_states.view(num_tokens, 1, k_in), ids, n_exp)
        inter = act_fn(gate_up.view(num_tokens * top_k, n2).to(hidden_states.dtype))
        out = mmq_moe(down_q, qt_down, inter.shape[1], inter.view(num_tokens, top_k, -1), ids, n_exp)
        return _weighted_sum(out, topk_weights, num_tokens, top_k).to(hidden_states.dtype)

    small = 2 <= num_tokens <= _MMVQ_MAX_TOKENS
    if (
        small and activation == "silu" and qt not in _OLD_MMVQ_GATE_UP_TYPES
        and _new_mmvq_serves(qt)
    ):
        from freetoken.kernel.ggml_mmq import GLU_SWIGLU, mmvq

        # one launch: silu(x @ gate) * (x @ up), gate rows then up rows in each slot
        i = n2 // 2
        k_in = hidden_states.shape[1]
        inter = mmvq(
            gate_up_q[:, i:], qt, k_in, hidden_states.view(num_tokens, 1, k_in),
            topk_ids.to(torch.int32), gate=gate_up_q[:, :i], glu_op=GLU_SWIGLU,
        ).view(num_tokens * top_k, i).to(hidden_states.dtype)
    else:
        # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
        gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
        inter = act_fn(gate_up)
    if small and _new_mmvq_serves(qt_down):
        from freetoken.kernel.ggml_mmq import mmvq

        out = mmvq(down_q, qt_down, inter.shape[1], inter.view(num_tokens, top_k, -1), topk_ids.to(torch.int32))
    else:
        # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
        out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt_down, h, num_tokens * top_k)
    return _weighted_sum(out.reshape(num_tokens, top_k, h), topk_weights, num_tokens, top_k).to(
        hidden_states.dtype
    )


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Upstream's Q4_0-only entry point: the general GGUF path pinned to Q4_0."""
    from freetoken.gguf_quant import GGUF_EXPERT_FORMATS

    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation,
        GGUF_EXPERT_FORMATS["q4_0"],
    )


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0"]
