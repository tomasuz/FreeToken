"""Qwen3.8-Flash-Next MTP (nextn) draft head, served from llama.cpp's separate MTP GGUF.

The head predicts token ``p+2`` from the base model's final wide residual at ``p`` (the
``hc_count`` streams before the output mixer) and the embedding of token ``p+1``, as
llama.cpp's ``graph_mtp`` does::

    h   = hnorm(R_p)                     per stream, (1+w) RMSNorm
    e   = enorm(embed(tok_{p+1}))        repeated per stream
    R   = eh_proj(cat[e, h])             per stream, 2H -> H
    R   = attention block + MoE block    one decoder layer over the streams
    out = lm_head(hc_head(R))            the base model's head

Two departures from the base layer, both measured against a reference implementation of
llama.cpp's graph over 744 greedy drafts of Qwen3.8 REAP-256 (acceptance 0.899 with full
context):

* Attention is dense over a ring of the last ``window`` positions the head has seen, not
  QSA over the whole sequence: the head barely uses context (0.884 at 64 positions, 0.816
  attending to itself only), and a small ring needs no KV pool, no page table and no prompt
  prefill over the whole prompt.
* The 512 routed experts (Q8_0, 5 MiB each) live in their own small offload cache: the head
  needs them (0.699 without), but not resident, and not competing with the base model's
  slots.

The head is kept out of the model's BaseOP tree (engine walks it for the base cache and
the base state dict) and loads its own weights; see ``Qwen4ExpForCausalLM.init_mtp``.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from freetoken.gguf_quant import GGML_F32, GGML_Q8_0, row_bytes
from freetoken.layers import BaseOP
from freetoken.layers.gguf import GGUFLinear
from freetoken.utils import init_logger

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)

_ARCH = "qwen4exp"
# positions the head's attention ring keeps (see the module docstring)
MTP_WINDOW = int(os.getenv("FREETOKEN_MTP_WINDOW", "64"))
# slots of the head's own expert cache (5 MiB each for Q8_0 experts)
MTP_CACHE_SLOTS = int(os.getenv("FREETOKEN_MTP_CACHE_SLOTS", "96"))


class _MTPAttention(Qwen4ExpAttention):
    """The base gated-GQA block over a dense attention ring instead of the QSA backend."""

    def __init__(self, config: "ModelConfig", window: int, *, prefix: str = "") -> None:
        super().__init__(config, layer_id=config.num_layers, prefix=prefix)
        del self.indexer  # QSA-only; the ring attends to every kept position
        self._window = window
        self._scale = config.attn_sm_scale or self.head_dim**-0.5
        self._k = None
        self._v = None
        self._pos = None

    def alloc_ring(self, device, dtype) -> None:
        shape = (self._window, self.num_kv, self.head_dim)
        self._k = torch.zeros(shape, device=device, dtype=dtype)
        self._v = torch.zeros(shape, device=device, dtype=dtype)
        self._pos = torch.full((self._window,), -1, device=device, dtype=torch.int64)

    def reset_ring(self) -> None:
        self._pos.fill_(-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        qg, k, v = self.qkv_proj.forward(x).split(self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.contiguous().view(-1, self.num_kv, self.head_dim)
        self.q_norm.forward_inplace(q)
        self.k_norm.forward_inplace(k)
        q, k = self.rotary.forward(positions, q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim))
        q = q.view(-1, self.num_q, self.head_dim)
        k = k.view(-1, self.num_kv, self.head_dim)
        v = v.contiguous().view(-1, self.num_kv, self.head_dim)
        # write this forward's rows first: a later row of the same forward sees the earlier
        # ones, a position already in the ring (a rejected draft's) is overwritten
        pos = positions.to(torch.int64)
        slots = pos % self._window
        self._k.index_copy_(0, slots, k.to(self._k.dtype))
        self._v.index_copy_(0, slots, v.to(self._v.dtype))
        self._pos.index_copy_(0, slots, pos)
        rep = self.num_q // self.num_kv
        kr = self._k.float().repeat_interleave(rep, dim=1)  # [W, HQ, D]
        vr = self._v.float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,whd->thw", q.float(), kr) * self._scale
        ring = self._pos.view(1, 1, -1)
        row = pos.view(-1, 1, 1)
        hidden = (ring < 0) | (ring > row) | (ring <= row - self._window)
        scores = scores.masked_fill(hidden, float("-inf"))
        o = torch.einsum("thw,whd->thd", scores.softmax(-1), vr).to(x.dtype)
        gated = o.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


def _swap_hc_to_gguf(hc: GatedResidual) -> None:
    """The MTP GGUF stores the hyper-connection projections as Q8_0, inject included."""
    width = hc.hidden_size * hc.hc_count
    if hc.use_combine:
        hc.input_mix_weight_down_block_inject = GGUFLinear(width, hc.lowrank + hc.hc_count, GGML_Q8_0)
    else:
        hc.input_mix_weight_down = GGUFLinear(width, hc.lowrank, GGML_Q8_0)
    hc.input_mix_weight_up = GGUFLinear(hc.lowrank, width, GGML_Q8_0)


class Qwen4ExpMTPHead(BaseOP):
    def __init__(self, config: "ModelConfig", num_experts: int, window: int = MTP_WINDOW) -> None:
        args = config.qwen4_args
        H = config.hidden_size
        self.hidden_size = H
        self.hc_count = args.hc_count
        eps = config.rms_norm_eps
        self.enorm = GroupedPlusOneRMSNorm(H, eps, 1)
        self.hnorm = GroupedPlusOneRMSNorm(H * args.hc_count, eps, args.hc_count)
        self.eh_proj = GGUFLinear(2 * H, H, GGML_Q8_0)
        self.attn_hyper_connection = GatedResidual(config)
        _swap_hc_to_gguf(self.attn_hyper_connection)
        self.self_attn = _MTPAttention(config, window)
        attn = self.self_attn
        attn.qkv_proj = GGUFLinear(H, sum(attn._qkv_split), GGML_Q8_0)
        attn.o_proj = GGUFLinear(attn.qo_attn_dim, H, GGML_Q8_0)
        self.mlp_hyper_connection = GatedResidual(config)
        _swap_hc_to_gguf(self.mlp_hyper_connection)
        mtp_cfg = dataclasses.replace(config, num_experts=num_experts)
        self.mlp = Qwen4ExpMoE(mtp_cfg, layer_id=0, prefix="mtp.mlp")
        sh = self.mlp.shared_expert
        inter = config.shared_expert_intermediate_size
        sh.gate_up_proj = GGUFLinear(H, 2 * inter, GGML_Q8_0)
        sh.down_proj = GGUFLinear(inter, H, GGML_Q8_0)
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        _swap_hc_to_gguf(self.hyper_connection_mixer)

    def forward(self, R_prev: torch.Tensor, next_ids: torch.Tensor, positions: torch.Tensor,
                embed, lm_head) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft logits for the token after each ``next_ids`` row, and the head's own wide
        residual (a chained draft's ``R_prev``). ``positions`` are those of ``next_ids``."""
        n, hc, H = R_prev.shape[0], self.hc_count, self.hidden_size
        h = self.hnorm.forward(R_prev).view(n, hc, H)
        e = self.enorm.forward(embed.forward(next_ids)).view(n, 1, H).expand(n, hc, H)
        R = self.eh_proj.forward(torch.cat([e, h], dim=-1).reshape(n * hc, 2 * H)).view(n, hc * H)
        x, inject = self.attn_hyper_connection.mix(R)
        R = self.attn_hyper_connection.combine(R, self.self_attn.forward(x, positions), inject)
        x, inject = self.mlp_hyper_connection.mix(R)
        R = self.mlp_hyper_connection.combine(R, self.mlp.forward(x), inject)
        y = self.hyper_connection_mixer.mix(R)[0]
        return lm_head.forward(y), R


# --------------------------------------------------------------------------------------
# Weights (llama.cpp MTP GGUF: blk.<num_layers>.* plus its own token_embd/output, unused)
# --------------------------------------------------------------------------------------


def _tensors(path: str) -> dict:
    import gguf

    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def _packed(t) -> torch.Tensor:
    shape = [int(x) for x in t.shape]
    rows = math.prod(shape[1:])
    return torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)).view(rows, -1)


def _dense(t, *, minus_one: bool = False) -> torch.Tensor:
    from freetoken.models.gguf.dequant import dequantize

    shape = [int(x) for x in reversed(t.shape)]
    w = dequantize(_packed(t).reshape(-1), int(t.tensor_type), torch.float32).reshape(shape)
    return w - 1 if minus_one else w


def mtp_state_dict(path: str, layer: int) -> dict[str, torch.Tensor]:
    """The head's state dict (keys relative to :class:`Qwen4ExpMTPHead`), expert banks aside."""
    T = _tensors(path)
    b = f"blk.{layer}."

    def q8(name):
        t = T[b + name]
        assert int(t.tensor_type) == GGML_Q8_0, (name, t.tensor_type)
        return _packed(t)

    sd = {
        "enorm.weight": _dense(T[b + "nextn.enorm.weight"], minus_one=True),
        "hnorm.weight": _dense(T[b + "nextn.hnorm.weight"], minus_one=True),
        "eh_proj.qweight": q8("nextn.eh_proj.weight"),
        "self_attn.qkv_proj.qweight": torch.cat(
            [q8("attn_q.weight"), q8("attn_k.weight"), q8("attn_v.weight")], dim=0
        ),
        "self_attn.o_proj.qweight": q8("attn_output.weight"),
        "self_attn.q_norm.weight": _dense(T[b + "attn_q_norm.weight"], minus_one=True),
        "self_attn.k_norm.weight": _dense(T[b + "attn_k_norm.weight"], minus_one=True),
        "mlp.gate.weight": _dense(T[b + "ffn_gate_inp.weight"]),
        "mlp.shared_expert.gate_up_proj.qweight": torch.cat(
            [q8("ffn_gate_shexp.weight"), q8("ffn_up_shexp.weight")], dim=0
        ),
        "mlp.shared_expert.down_proj.qweight": q8("ffn_down_shexp.weight"),
        "mlp.shared_expert_gate.weight": _dense(T[b + "ffn_gate_inp_shexp.weight"]).reshape(1, -1),
        "hyper_connection_mixer.hc_norm.weight": _dense(T[b + "nextn.hc_head_norm.weight"], minus_one=True),
        "hyper_connection_mixer.input_mix_weight_down.qweight": q8("nextn.hc_head_down.weight"),
        "hyper_connection_mixer.input_mix_weight_up.qweight": q8("nextn.hc_head_up.weight"),
    }
    for mod, part in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        sd[f"{mod}.hc_norm.weight"] = _dense(T[b + f"{part}_norm.weight"], minus_one=True)
        sd[f"{mod}.input_mix_weight_down_block_inject.qweight"] = torch.cat(
            [q8(f"{part}_down.weight"), q8(f"{part}_inject.weight")], dim=0
        )
        sd[f"{mod}.input_mix_weight_up.qweight"] = q8(f"{part}_up.weight")
    return sd


def mtp_layer_and_experts(path: str) -> tuple[int, int, tuple[int, int]]:
    """``(layer index in the file, routed experts, (gate_up type, down type))``."""
    import gguf

    r = gguf.GGUFReader(path)
    meta = {k: f.contents() for k, f in r.fields.items() if k.startswith(_ARCH + ".")}
    layer = int(meta[f"{_ARCH}.block_count"]) - int(meta.get(f"{_ARCH}.nextn_predict_layers", 1))
    experts = int(meta[f"{_ARCH}.expert_count"])
    names = {t.name: int(t.tensor_type) for t in r.tensors}
    gate, up, down = (names[f"blk.{layer}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))
    assert gate == up, "gate and up share one bank"
    return layer, experts, (gate, down)


def mtp_expert_banks(path: str, layer: int, experts: int, inter: int, hidden: int,
                     types: tuple[int, int]) -> dict[str, list[torch.Tensor]]:
    """Pinned host banks ``gate_up [E, 2I, row_bytes(H)]`` and ``down [E, H, row_bytes(I)]``."""
    T = _tensors(path)
    b = f"blk.{layer}."
    gate_up = torch.empty((experts, 2 * inter, row_bytes(hidden, types[0])), dtype=torch.uint8)
    down = torch.empty((experts, hidden, row_bytes(inter, types[1])), dtype=torch.uint8)
    gate_up[:, :inter].copy_(_packed(T[b + "ffn_gate_exps.weight"]).view(experts, inter, -1))
    gate_up[:, inter:].copy_(_packed(T[b + "ffn_up_exps.weight"]).view(experts, inter, -1))
    down.copy_(_packed(T[b + "ffn_down_exps.weight"]).view(experts, hidden, -1))
    return {"gate_up": [gate_up.pin_memory()], "down": [down.pin_memory()]}


def build_mtp_cache(device, experts: int, types: tuple[int, int], banks, slots: int) -> "OffloadMoeCache":
    """A one-layer offload cache over the head's experts."""
    from freetoken.moe.host_banks import HostResidency
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=experts,
        cache_size=slots,
        device=device,
        cache_policy="lru",
        prefill_overlap=False,
        prefill_hit_d2d=False,
        quant_format="gguf",
        # only ever filled through ensure_experts, a few rows at a time
        slot_floor=min(slots, experts),
    )
    cache.gguf_layer_types = [types]
    cache.set_bank_sources(banks, layer_residency=[HostResidency.PINNED.value])
    return cache


__all__ = [
    "MTP_CACHE_SLOTS",
    "MTP_WINDOW",
    "Qwen4ExpMTPHead",
    "build_mtp_cache",
    "mtp_expert_banks",
    "mtp_layer_and_experts",
    "mtp_state_dict",
]
