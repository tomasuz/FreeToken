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
# experts per draft row: the head's routing is flat enough that its top 4 keep the
# acceptance (0.876 vs 0.878 at top 10 on Qwen3.8) at 40% of the expert traffic
MTP_TOPK = int(os.getenv("FREETOKEN_MTP_TOPK", "4"))


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


# Each projection of the head: its module path and the GGUF tensors concatenated (in row
# order) into its weight. Their ggml type comes from the file (llama.cpp's MTP GGUFs have
# been Q8_0, but nothing requires it): a type the GGUF kernels serve stays packed, anything
# else -- f16/bf16/f32, an i-quant, parts of different types -- is dequantized to a dense
# bf16 weight, which for this one layer costs little memory.
_SITES: dict[str, tuple[str, ...]] = {
    "eh_proj": ("nextn.eh_proj.weight",),
    "self_attn.qkv_proj": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "self_attn.o_proj": ("attn_output.weight",),
    "mlp.shared_expert.gate_up_proj": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "mlp.shared_expert.down_proj": ("ffn_down_shexp.weight",),
    "attn_hyper_connection.input_mix_weight_down_block_inject": ("hc_attn_down.weight", "hc_attn_inject.weight"),
    "attn_hyper_connection.input_mix_weight_up": ("hc_attn_up.weight",),
    "mlp_hyper_connection.input_mix_weight_down_block_inject": ("hc_ffn_down.weight", "hc_ffn_inject.weight"),
    "mlp_hyper_connection.input_mix_weight_up": ("hc_ffn_up.weight",),
    "hyper_connection_mixer.input_mix_weight_down": ("nextn.hc_head_down.weight",),
    "hyper_connection_mixer.input_mix_weight_up": ("nextn.hc_head_up.weight",),
}
# the historical layout: every projection Q8_0
Q8_SITES: dict[str, int | None] = {site: GGML_Q8_0 for site in _SITES}


def mtp_site_types(tensor_types: dict[str, int], layer: int) -> dict[str, int | None]:
    """Per projection: the ggml type it stays packed in, or None for a dense bf16 weight.
    ``tensor_types`` maps GGUF tensor names to their ggml type."""
    from freetoken.layers.gguf import _QUANTS

    out: dict[str, int | None] = {}
    for site, names in _SITES.items():
        kinds = {tensor_types[f"blk.{layer}.{n}"] for n in names}
        kind = kinds.pop() if len(kinds) == 1 else None
        out[site] = kind if kind in _QUANTS else None
    return out


class _DenseLinear(BaseOP):
    """A plain bf16 projection for a head weight the GGUF kernels do not serve packed."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def _linear(in_features: int, out_features: int, kind: int | None) -> BaseOP:
    return GGUFLinear(in_features, out_features, kind) if kind is not None else _DenseLinear(in_features, out_features)


def _swap_hc_to_gguf(hc: GatedResidual, prefix: str, types: dict[str, int | None]) -> None:
    """The MTP GGUF stores the hyper-connection projections quantized, inject included."""
    width = hc.hidden_size * hc.hc_count
    if hc.use_combine:
        site = f"{prefix}.input_mix_weight_down_block_inject"
        hc.input_mix_weight_down_block_inject = _linear(width, hc.lowrank + hc.hc_count, types[site])
    else:
        hc.input_mix_weight_down = _linear(width, hc.lowrank, types[f"{prefix}.input_mix_weight_down"])
    hc.input_mix_weight_up = _linear(hc.lowrank, width, types[f"{prefix}.input_mix_weight_up"])


class Qwen4ExpMTPHead(BaseOP):
    def __init__(self, config: "ModelConfig", num_experts: int, window: int = MTP_WINDOW,
                 site_types: dict[str, int | None] | None = None) -> None:
        types = site_types or Q8_SITES
        args = config.qwen4_args
        H = config.hidden_size
        self.hidden_size = H
        self.hc_count = args.hc_count
        eps = config.rms_norm_eps
        self.enorm = GroupedPlusOneRMSNorm(H, eps, 1)
        self.hnorm = GroupedPlusOneRMSNorm(H * args.hc_count, eps, args.hc_count)
        self.eh_proj = _linear(2 * H, H, types["eh_proj"])
        self.attn_hyper_connection = GatedResidual(config)
        _swap_hc_to_gguf(self.attn_hyper_connection, "attn_hyper_connection", types)
        self.self_attn = _MTPAttention(config, window)
        attn = self.self_attn
        attn.qkv_proj = _linear(H, sum(attn._qkv_split), types["self_attn.qkv_proj"])
        attn.o_proj = _linear(attn.qo_attn_dim, H, types["self_attn.o_proj"])
        self.mlp_hyper_connection = GatedResidual(config)
        _swap_hc_to_gguf(self.mlp_hyper_connection, "mlp_hyper_connection", types)
        mtp_cfg = dataclasses.replace(config, num_experts=num_experts, num_experts_per_tok=min(MTP_TOPK, config.num_experts_per_tok))
        self.mlp = Qwen4ExpMoE(mtp_cfg, layer_id=0, prefix="mtp.mlp")
        sh = self.mlp.shared_expert
        inter = config.shared_expert_intermediate_size
        sh.gate_up_proj = _linear(H, 2 * inter, types["mlp.shared_expert.gate_up_proj"])
        sh.down_proj = _linear(inter, H, types["mlp.shared_expert.down_proj"])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        _swap_hc_to_gguf(self.hyper_connection_mixer, "hyper_connection_mixer", types)

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


def _dense_any(t) -> torch.Tensor:
    """A GGUF tensor as f32 ``[rows, cols]``: on the CPU where the reference dequantizers
    cover the type, else through the GPU kernels (the i-quants)."""
    try:
        return _dense(t)
    except NotImplementedError:
        from freetoken.kernel.gguf import ggml_dequantize

        rows, cols = int(math.prod(int(x) for x in t.shape[1:])), int(t.shape[0])
        return ggml_dequantize(_packed(t).cuda(), int(t.tensor_type), rows, cols, torch.float32).cpu()


def mtp_state_dict(path: str, layer: int, site_types: dict[str, int | None] | None = None) -> dict[str, torch.Tensor]:
    """The head's state dict (keys relative to :class:`Qwen4ExpMTPHead`), expert banks aside.
    ``site_types`` as :func:`mtp_site_types` gives it (default: every projection Q8_0)."""
    T = _tensors(path)
    b = f"blk.{layer}."
    types = site_types or Q8_SITES

    sd = {
        "enorm.weight": _dense(T[b + "nextn.enorm.weight"], minus_one=True),
        "hnorm.weight": _dense(T[b + "nextn.hnorm.weight"], minus_one=True),
        "self_attn.q_norm.weight": _dense(T[b + "attn_q_norm.weight"], minus_one=True),
        "self_attn.k_norm.weight": _dense(T[b + "attn_k_norm.weight"], minus_one=True),
        "mlp.gate.weight": _dense(T[b + "ffn_gate_inp.weight"]),
        "mlp.shared_expert_gate.weight": _dense(T[b + "ffn_gate_inp_shexp.weight"]).reshape(1, -1),
        "hyper_connection_mixer.hc_norm.weight": _dense(T[b + "nextn.hc_head_norm.weight"], minus_one=True),
    }
    for mod, part in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        sd[f"{mod}.hc_norm.weight"] = _dense(T[b + f"{part}_norm.weight"], minus_one=True)
    for site, names in _SITES.items():
        parts = [T[b + n] for n in names]
        kind = types[site]
        if kind is not None:
            for n, t in zip(names, parts):
                assert int(t.tensor_type) == kind, (n, int(t.tensor_type), kind)
            sd[f"{site}.qweight"] = torch.cat([_packed(t) for t in parts], dim=0)
        else:
            sd[f"{site}.weight"] = torch.cat([_dense_any(t) for t in parts], dim=0)
    return sd


def mtp_tensor_types(path: str) -> dict[str, int]:
    """Every tensor of the MTP GGUF and its ggml type."""
    return {name: int(t.tensor_type) for name, t in _tensors(path).items()}


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
    "Q8_SITES",
    "MTP_CACHE_SLOTS",
    "MTP_TOPK",
    "MTP_WINDOW",
    "Qwen4ExpMTPHead",
    "build_mtp_cache",
    "mtp_expert_banks",
    "mtp_layer_and_experts",
    "mtp_site_types",
    "mtp_state_dict",
    "mtp_tensor_types",
]
