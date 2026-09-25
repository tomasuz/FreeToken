"""Qwen3.8-Flash-Next GGUF adapter (llama.cpp ``qwen4exp`` exports, e.g. unsloth's UD quants).

Serves the same GGUF file llama.cpp serves, so a model already on disk for llama.cpp needs no
second download. Two halves:

* :func:`parse_gguf_config` -- the GGUF KV metadata turned into the attribute set the HF
  config carries, then run through the one :func:`~.config.parse_config`, so a GGUF and its
  safetensors origin produce the same :class:`ModelConfig`. The few facts llama.cpp's
  qwen4exp graph hard-codes instead of storing (sigmoid GDN output gate, silu MoE, the
  indexer's single key head) are pinned here and cross-checked against the tensor shapes.
* :func:`iter_gguf_weights` -- every dense (non-expert) tensor under the FreeToken state-dict
  name, block-quantized projections kept packed for the GGUF kernels and the rest in bf16, with llama.cpp's conversion (``conversion/qwen.py``
  ``Qwen3NextModel`` / ``_LinearAttentionVReorderBase`` and ``conversion/qwen4exp.py``)
  undone: the GDN value heads go back from ggml's tiled order to HF's grouped one,
  ``ssm_a = -exp(A_log)`` is inverted, the ``+1`` baked into the zero-centred norms is taken
  off again, and split projections are re-fused the way the safetensors reader fuses them.

The routed experts and the PLE n-gram table are not state-dict tensors: the experts stream
from the offload cache's expert banks and the table is read row by row (``ple_disk``).
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers import BaseOP
from freetoken.models.gguf.dequant import GGML_F32, GGML_NAME, dequantize

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"

# What llama.cpp's qwen4exp graph fixes rather than reads from the file.
_HIDDEN_ACT = "silu"
# src/models/qwen4exp.cpp: "the one numerical difference from Qwen3.5's GDN: sigmoid output gate"
_GDN_OUTPUT_GATE = "sigmoid"
# The GGUF stores the resolved n-gram vocab sizes, not the base they were derived from. The
# config field only feeds dummy-weight runs (model.load_host_tables re-derives the constants),
# where _ngram_vocab_base() recovers a base that reproduces the stored sizes.
_MAKE_NGRAM_VOCAB_DIVISIBLE_BY = 128


def _meta(shim: "GgufConfigShim"):
    m = shim.metadata

    def g(key: str, default=None):
        val = m.get(f"{_ARCH}.{key}", default)
        if val is None:
            raise KeyError(f"missing GGUF metadata key {_ARCH}.{key}")
        return val

    return g


def _f32(v) -> float:
    """A float32 KV value as the decimal it was written from: GGUF stores 1e-6 as
    9.999999974752427e-07, and the HF config (and every comparison against it) says 1e-06."""
    return float(f"{float(v):.7g}")


def _prev_prime_plus_one(n: int) -> int:
    """Smallest ``b`` whose first prime at or above ``b`` is ``n`` (``n`` prime): the ``ngram_vocab_size_base`` that reproduces head 0's vocab size."""
    from .ple import _is_prime

    k = n - 1
    while k > 1 and not _is_prime(k):
        k -= 1
    return k + 1


def _hf_like_config(shim: "GgufConfigShim") -> types.SimpleNamespace:
    """The GGUF metadata as the attributes :func:`parse_config` reads off a HF config."""
    from freetoken.models.gguf.reader import gguf_tensor_location

    g = _meta(shim)
    n_layers = int(g("block_count"))
    head_dim = int(g("attention.key_length"))
    if int(g("attention.value_length")) != head_dim:
        raise ValueError("qwen4exp GGUF: key and value head dims differ")

    # A full-attention layer is a compressed (QSA) one; every other is a GDN layer.
    ratios = [int(r) for r in g("attention.compress_ratios")][:n_layers]
    layer_types = ["full_attention" if r else "linear_attention" for r in ratios]
    full_ratios = {r for r in ratios if r}
    if len(full_ratios) != 1:
        raise ValueError(f"qwen4exp GGUF: expected one indexer compress ratio, got {sorted(full_ratios)}")
    interval = int(g("full_attention_interval"))
    expect = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)]
    if layer_types != expect:
        raise ValueError("qwen4exp GGUF: compress_ratios disagree with full_attention_interval")

    num_v_heads = int(g("ssm.time_step_rank"))
    inner = int(g("ssm.inner_size"))
    if inner % num_v_heads:
        raise ValueError(f"qwen4exp GGUF: ssm.inner_size {inner} not a multiple of {num_v_heads} value heads")

    # The indexer's key head count is implied by its key projection (llama.cpp splits
    # index_qk_proj into q and k at conversion and keeps no count).
    index_head_dim = int(g("attention.indexer.key_length"))
    full0 = layer_types.index("full_attention")
    k_rows = gguf_tensor_location(shim.model_path, f"blk.{full0}.indexer.k_proj.weight").shape[0]
    if k_rows % index_head_dim:
        raise ValueError(f"qwen4exp GGUF: indexer k_proj has {k_rows} rows, not a multiple of {index_head_dim}")

    ngram_size = int(g("ple.ngram_size"))
    heads_per_ngram = int(g("ple.heads_per_ngram"))
    row_dim = int(g("embedding_length_per_layer_input"))
    head_vocab_sizes = [int(x) for x in g("ple.head_vocab_sizes")]
    eos = int(g("ple.eos_token_id"))

    text = types.SimpleNamespace(
        num_hidden_layers=n_layers,
        hidden_size=int(g("embedding_length")),
        num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")),
        head_dim=head_dim,
        rope_parameters={
            "rope_theta": _f32(g("rope.freq_base")),
            "partial_rotary_factor": int(g("rope.dimension_count")) / head_dim,
            # served text-only: the interleaved mRoPE sections reduce to plain partial rope
            "rope_type": "default",
        },
        max_position_embeddings=int(g("context_length")),
        layer_types=layer_types,
        full_attention_interval=interval,
        rms_norm_eps=_f32(g("attention.layer_norm_rms_epsilon")),
        hidden_act=_HIDDEN_ACT,
        output_gate_type=_GDN_OUTPUT_GATE,
        linear_num_key_heads=int(g("ssm.group_count")),
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=int(g("ssm.state_size")),
        linear_value_head_dim=inner // num_v_heads,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")),
        indexer_head_dim=index_head_dim,
        indexer_n_heads=int(g("attention.indexer.head_count")),
        indexer_kv_heads=k_rows // index_head_dim,
        indexer_budget=int(g("attention.indexer.top_k")),
        indexer_compress_ratio=full_ratios.pop(),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        intermediate_size=0,
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        eos_token_id=eos,
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        # parse_config wants HF's one-based ids; the GGUF stores them zero-based
        ple_layer_ids=[int(i) + 1 for i in g("ple.layers")],
        ple_embed_dim=(ngram_size - 1) * heads_per_ngram * row_dim,
        ple_conv_kernel_size=int(g("ple.conv_kernel")),
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=_prev_prime_plus_one(head_vocab_sizes[0]),
        make_ngram_vocab_size_divisible_by=_MAKE_NGRAM_VOCAB_DIVISIBLE_BY,
        # the GGUF holds the table as one tensor, not HF's row blocks
        split_ngram_parts=1,
    )
    return types.SimpleNamespace(
        text_config=text,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        image_token_id=shim.metadata.get(f"{_ARCH}.ple.image_token_id"),
    )


def _expert_types(model_path: str, num_layers: int) -> tuple[tuple[int, int], ...]:
    """Per layer ``(gate_up type, down type)`` of the routed experts. The ggml type is per
    tensor, and unsloth's UD quants vary it by layer; gate and up must agree (one bank)."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    types = {t.name: int(t.tensor_type) for p in gguf_shard_paths(model_path) for t in _reader(p).tensors}
    out = []
    for layer in range(num_layers):
        gate, up, down = (types.get(f"blk.{layer}.{sfx}") for sfx in _EXPERT_SUFFIXES)
        if None in (gate, up, down):
            raise ValueError(f"qwen4exp GGUF: layer {layer} lacks routed-expert tensors")
        if gate != up:
            raise ValueError(
                f"qwen4exp GGUF: layer {layer} gate is {GGML_NAME.get(gate, gate)} but up is "
                f"{GGML_NAME.get(up, up)}; they share one bank, so they must match"
            )
        out.append((gate, down))
    return tuple(out)


def parse_gguf_config(shim: "GgufConfigShim") -> "ModelConfig":
    import dataclasses

    from .config import parse_config

    cfg = parse_config(_hf_like_config(shim))
    # The routed experts stream from the offload cache as their native ggml bytes, each
    # layer at its own type; the dense weights stay packed where a GGUF kernel reads their
    # type (see _packed_type) and are dequantized to bf16 otherwise.
    return dataclasses.replace(
        cfg,
        expert_quant="gguf",
        gguf_expert_types=_expert_types(shim.model_path, cfg.num_layers),
        gguf_dense_types=_dense_types(shim.model_path),
    )


def _dense_types(model_path: str) -> tuple[tuple[str, int], ...]:
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    return tuple(
        (t.name, int(t.tensor_type))
        for p in gguf_shard_paths(model_path)
        for t in _reader(p).tensors
        if not t.name.endswith(_EXPERT_SUFFIXES) and t.name != PLE_TABLE_TENSOR
    )


# --------------------------------------------------------------------------------------
# Routed experts: per-layer host banks of native ggml bytes, for the offload cache
# --------------------------------------------------------------------------------------


def _expert_bank_specs(cfg: "ModelConfig") -> list[dict]:
    from freetoken.gguf_quant import row_bytes

    E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
    return [
        {
            "gate_up": ((E, 2 * I, row_bytes(H, gate_up)), torch.uint8),
            "down": ((E, H, row_bytes(I, down)), torch.uint8),
        }
        for gate_up, down in cfg.gguf_expert_types
    ]


def load_q4_0_expert_sources(model_path: str, config: "ModelConfig", *, layer_sink=None) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' packed ggml bytes (the name is the loader
    hook's, from when every GGUF expert was Q4_0).

    ``gate_up`` is one ``[E, 2I, row_bytes(H)]`` tensor per layer, gate rows then up rows of
    each expert -- the order the MoE kernel's activation splits; ``down`` is
    ``[E, H, row_bytes(I)]``. Rows are copied verbatim from the file (no dequant), each
    layer at its own ggml type (``config.gguf_expert_types``). ``layer_sink`` as in
    :func:`freetoken.models.gemma4.gguf.load_q4_0_expert_sources`.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_varying_layer_banks

    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4exp GGUF expert banks support TP=1 only")
    L, E, I = config.num_layers, config.num_experts, config.moe_intermediate_size
    hb = alloc_varying_layer_banks(_expert_bank_specs(config))  # lazy anon mmaps (unpinned)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen: dict[str, set[int]] = {sfx: set() for sfx in _EXPERT_SUFFIXES}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None  # gate, up, down
        for t in iter_gguf_tensors(model_path):
            sfx = t.name.split(".", 2)[2] if t.name.startswith("blk.") else None
            if sfx not in seen:
                continue
            layer = int(t.name.split(".")[1])
            gate_up_type, down_type = config.gguf_expert_types[layer]
            if sfx == "ffn_down_exps.weight":
                assert t.ggml_type == down_type, (t.name, t.ggml_type, down_type)
                bank = banks["down"][layer]
                bank.copy_(t.packed().reshape(bank.shape))
            else:
                assert t.ggml_type == gate_up_type, (t.name, t.ggml_type, gate_up_type)
                bank = banks["gate_up"][layer]
                half = slice(0, I) if sfx == "ffn_gate_exps.weight" else slice(I, 2 * I)
                bank[:, half].copy_(t.packed().reshape(E, I, bank.shape[-1]))
            seen[sfx].add(layer)
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(range(L))
    missing = {sfx: sorted(want - got) for sfx, got in seen.items() if got != want}
    assert not missing, f"qwen4exp GGUF: missing routed-expert layers {missing}"
    return banks


def dummy_q4_0_expert_sources(config: "ModelConfig") -> dict[str, list[torch.Tensor]]:
    """Zeroed banks shaped like :func:`load_q4_0_expert_sources` (a zero block scale is a
    valid all-zero block in every ggml type, where random bytes could decode to NaN)."""
    from freetoken.moe.host_banks import alloc_varying_layer_banks, pin_banks

    hb = alloc_varying_layer_banks(_expert_bank_specs(config))
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.zero_()
    if torch.cuda.is_available():
        pin_banks(hb)
    return banks


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------


def _reorder_v_heads(t: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """llama.cpp ``_LinearAttentionVReorderBase._reorder_v_heads``, verbatim: views ``dim`` as
    ``[num_k_heads, num_v_per_k, head_dim]`` and swaps the first two."""
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


class _VHeads:
    """Undo the GDN value-head reorder. The forward map views the heads as ``[K, R]`` and
    swaps to ``[R, K]``; running the same map with the roles swapped swaps them back."""

    def __init__(self, cfg: "ModelConfig") -> None:
        g = cfg.attention_groups
        lin = next(x for x in g if x.name == "linear")
        self.k_heads = lin.num_key_heads
        self.v_per_k = lin.num_value_heads // lin.num_key_heads
        self.key_dim = lin.key_head_dim
        self.value_dim = lin.value_head_dim
        self.active = lin.num_key_heads != lin.num_value_heads

    def untile(self, t: torch.Tensor, dim: int, head_dim: int) -> torch.Tensor:
        if not self.active:
            return t
        return _reorder_v_heads(t, dim, self.v_per_k, self.k_heads, head_dim)

    def untile_vec(self, t: torch.Tensor) -> torch.Tensor:
        """A per-value-head vector (``A_log``, ``dt_bias``)."""
        return self.untile(t.unsqueeze(-1), 0, 1).squeeze(-1)


def _to_bf16(t) -> torch.Tensor:
    """Dequantize; an F32 source stays F32 (A_log, dt_bias, norms, router) -- the loader
    casts to each parameter's own dtype, so narrowing here would only throw precision away."""
    dtype = torch.float32 if t.ggml_type == GGML_F32 else torch.bfloat16
    return dequantize(t.packed().reshape(-1), t.ggml_type, dtype).reshape(t.shape)


def _to_f32(t) -> torch.Tensor:
    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32).reshape(t.shape)


# Norms the conversion stored as ``1 + w`` (Qwen3NextModel's ``norm.weight`` rule, and
# qwen4exp's explicit ones for the PLE and indexer norms). FreeToken applies ``1 + w`` itself,
# at runtime in fp32 (weight._ZERO_CENTERED_NORM_SUFFIXES), so the raw ``w`` is what it loads.
_PLUS_ONE = {
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
}
# 1:1 renames, no transform.
_PLAIN = {
    "attn_output.weight": "self_attn.o_proj.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",  # the GDN gated norm: plain w*x, no +1
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
}
# Parts re-fused into one buffer: fused name -> ordered GGUF parts.
_FUSED = {
    "self_attn.qkv_proj.weight": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "self_attn.indexer.index_qk_proj.weight": ("indexer.q_proj.weight", "indexer.k_proj.weight"),
    "mlp.shared_expert.gate_up_proj.weight": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    # the safetensors reader's in_proj order (register._QWEN3_5_PACKED): qkv | z | b | a
    "linear_attn.in_proj.weight": ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"),
    # low-rank down | block inject, zero-padded to a multiple of 16 rows (weight._PAD_TO)
    "attn_hyper_connection.input_mix_weight_down_block_inject.weight": ("hc_attn_down.weight", "hc_attn_inject.weight"),
    "mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ("hc_ffn_down.weight", "hc_ffn_inject.weight"),
}
_PART_OF = {part: (fused, idx) for fused, parts in _FUSED.items() for idx, part in enumerate(parts)}
_PAD_ROWS_TO = {"input_mix_weight_down_block_inject.weight": 16}
# Routed experts: the offload cache's expert banks, not the state dict.
_EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
# The n-gram table: read row by row from the file (ple_disk), never materialized.
PLE_TABLE_TENSOR = "per_layer_token_embd.weight"
_TOP_LEVEL = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_hc_down.weight": "model.hyper_connection_mixer.input_mix_weight_down.weight",
    "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight",
}
_TOP_LEVEL_PLUS_ONE = {"output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight"}



# --------------------------------------------------------------------------------------
# Dense weights kept packed
# --------------------------------------------------------------------------------------
# A block-quantized projection stays in its GGUF layout (a GGUFLinear) instead of being
# dequantized to bf16: half the VRAM, which the offload cache turns into expert slots, and
# half the bytes each decode step reads. Keyed by module path under ``model.layers.N``,
# parts in row order; a module is packed only when all its parts share one kernel type.
_PACKED_LINEAR = {
    "self_attn.qkv_proj": ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
    "self_attn.o_proj": ("attn_output.weight",),
    "linear_attn.out_proj": ("ssm_out.weight",),
    "mlp.shared_expert.gate_up_proj": ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    "mlp.shared_expert.down_proj": ("ffn_down_shexp.weight",),
    "attn_hyper_connection.input_mix_weight_up": ("hc_attn_up.weight",),
    "mlp_hyper_connection.input_mix_weight_up": ("hc_ffn_up.weight",),
    "ple.key_proj": ("ple_key.weight",),
    "ple.value_proj": ("ple_value.weight",),
}
# Fusions of a packable head and an unquantized tail (F32 in the GGUF): the head becomes a
# GGUFLinear, the few tail rows a dense matrix, and the op concatenates the two outputs.
_PACKED_SPLIT = {
    "linear_attn.in_proj": (("attn_qkv.weight", "attn_gate.weight"), ("ssm_beta.weight", "ssm_alpha.weight")),
    "attn_hyper_connection.input_mix_weight_down_block_inject": (("hc_attn_down.weight",), ("hc_attn_inject.weight",)),
    "mlp_hyper_connection.input_mix_weight_down_block_inject": (("hc_ffn_down.weight",), ("hc_ffn_inject.weight",)),
}
_PACKED_TOP = {
    "model.hyper_connection_mixer.input_mix_weight_down": "output_hc_down.weight",
    "model.hyper_connection_mixer.input_mix_weight_up": "output_hc_up.weight",
    "model.embed_tokens": "token_embd.weight",
    "lm_head": "output.weight",
}


def _packed_type(types: dict[str, int], names) -> int | None:
    """The one GGUF-kernel type all ``names`` share, or None (missing, mixed, unquantized)."""
    from freetoken.layers.gguf import _QUANTS

    found = {types.get(n) for n in names}
    if len(found) != 1:
        return None
    t = found.pop()
    return t if t in _QUANTS else None


def packing_plan(cfg: "ModelConfig") -> dict[str, int]:
    """``{module path: ggml type}`` of every module whose weight stays packed."""
    if not cfg.gguf_dense_types:
        return {}
    types = dict(cfg.gguf_dense_types)
    plan: dict[str, int] = {}
    for layer in range(cfg.num_layers):
        for mod, parts in _PACKED_LINEAR.items():
            t = _packed_type(types, [f"blk.{layer}.{p}" for p in parts])
            if t is not None:
                plan[f"model.layers.{layer}.{mod}"] = t
        for mod, (head, _tail) in _PACKED_SPLIT.items():
            t = _packed_type(types, [f"blk.{layer}.{p}" for p in head])
            if t is not None:
                plan[f"model.layers.{layer}.{mod}"] = t
    for mod, name in _PACKED_TOP.items():
        t = _packed_type(types, [name])
        if t is not None:
            plan[mod] = t
    return plan


# Up to this many tokens the dense tail is a multiply-reduce (the decode and MTP-verify sizes).
_SMALL_BATCH = 8


class PackedSplitLinear(BaseOP):
    """``cat([packed(x), x @ dense.T], -1)``: a GGUF block-quantized head over most output
    rows and a small dense tail for the rows the GGUF keeps unquantized."""

    def __init__(self, in_features: int, packed_rows: int, dense_rows: int, quant_type: int):
        from freetoken.layers.gguf import GGUFLinear

        self.packed = GGUFLinear(in_features, packed_rows, quant_type)
        self.dense = torch.empty(dense_rows, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= _SMALL_BATCH:
            # a handful of output rows is a poor GEMM shape for rocBLAS: 45 us for the 4
            # block-inject rows of a 10240-wide input, 9 us as a broadcast multiply-reduce
            tail = (x.unsqueeze(1) * self.dense).sum(-1)
        else:
            tail = torch.nn.functional.linear(x, self.dense)
        return torch.cat([self.packed.forward(x), tail], dim=-1)


class GGUFLMHead(BaseOP):
    """Untied LM head over a packed GGUF ``output.weight`` (``ParallelLMHead``'s contract, TP=1)."""

    def __init__(self, vocab_size: int, hidden_size: int, quant_type: int):
        from freetoken.layers.gguf import GGUFLinear

        self.head = GGUFLinear(hidden_size, vocab_size, quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx

        batch = get_global_ctx().batch
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return self.head.forward(x)


def _split_tail_rows(cfg: "ModelConfig", mod: str) -> int:
    if mod == "linear_attn.in_proj":
        return 2 * cfg.linear_attention_group().num_value_heads  # beta | alpha
    return cfg.qwen4_args.hc_count  # the block-inject rows


def convert_qwen4exp_to_gguf(model, cfg: "ModelConfig") -> None:
    """In place: swap every module of :func:`packing_plan` for its packed GGUF op."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    plan = packing_plan(cfg)
    layers = model.model.layers.op_list
    for path, qt in plan.items():
        parts = path.split(".")
        if parts[:2] == ["model", "layers"]:
            owner, rel = layers[int(parts[2])], parts[3:]
        else:
            owner, rel = model, parts
        for attr in rel[:-1]:
            owner = getattr(owner, attr)
        attr = rel[-1]
        rel_path = ".".join(rel)
        if path == "model.embed_tokens":
            new = GGUFEmbedding(cfg.vocab_size, cfg.hidden_size, qt)
        elif path == "lm_head":
            assert not cfg.tie_word_embeddings, "qwen4exp GGUF: a tied head has no output.weight"
            new = GGUFLMHead(cfg.vocab_size, cfg.hidden_size, qt)
        else:
            out_features, in_features = getattr(owner, attr).weight.shape
            if rel_path in _PACKED_SPLIT:
                tail = _split_tail_rows(cfg, rel_path)
                # the hc fusion's zero pad rows (16-row GEMM alignment) are not carried over
                head = out_features - tail if rel_path == "linear_attn.in_proj" else cfg.qwen4_args.hc_lowrank
                new = PackedSplitLinear(in_features, head, tail, qt)
            else:
                new = GGUFLinear(in_features, out_features, qt)
        setattr(owner, attr, new)


def _transform(suffix: str, t, vh: _VHeads) -> torch.Tensor:
    """One GGUF part (``suffix`` after ``blk.N.``) back in HF layout, bf16."""
    if suffix == "attn_qkv.weight":
        w = _to_bf16(t)
        qk = 2 * vh.k_heads * vh.key_dim
        return torch.cat([w[:qk], vh.untile(w[qk:], 0, vh.value_dim)], dim=0)
    if suffix == "attn_gate.weight":
        return vh.untile(_to_bf16(t), 0, vh.value_dim)
    if suffix in ("ssm_beta.weight", "ssm_alpha.weight"):
        return vh.untile(_to_bf16(t), 0, 1)
    return _to_bf16(t)



def _packed_rows(suffix: str, t, vh: _VHeads) -> torch.Tensor:
    """One GGUF part as its packed ``[rows, row_bytes]`` bytes, back in HF layout. The value
    head untiling moves whole rows (output features), or for ``ssm_out`` whole head-wide
    column runs, which are whole quant blocks, so no block is ever split."""
    from freetoken.gguf_quant import BLOCK_SHAPE

    w = t.packed()
    if suffix == "attn_qkv.weight":
        qk = 2 * vh.k_heads * vh.key_dim
        return torch.cat([w[:qk], vh.untile(w[qk:], 0, vh.value_dim)], dim=0)
    if suffix == "attn_gate.weight":
        return vh.untile(w, 0, vh.value_dim)
    if suffix == "ssm_out.weight":
        block, size = BLOCK_SHAPE[t.ggml_type]
        assert vh.value_dim % block == 0, (vh.value_dim, block)
        return vh.untile(w, 1, vh.value_dim // block * size)
    return w


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(state_dict_name, tensor)`` for every non-expert Qwen3.8 parameter.

    Block-quantized dense weights a GGUF kernel reads stay packed as ``.qweight`` (see
    :func:`packing_plan`); the rest are dequantized to bf16, F32 ones stay F32. The routed
    experts stay packed in the offload cache's banks and the PLE table on disk.
    """
    from freetoken.distributed import get_tp_info
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata
    from freetoken.utils import cached_load_hf_config

    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4exp GGUF weight loading supports TP=1 only")
    assert not include_moe_experts, (
        "qwen4exp GGUF routed experts are served from the offload cache's expert banks"
    )
    if not include_non_moe:
        return

    cfg = parse_gguf_config(cached_load_hf_config(model_path))
    vh = _VHeads(cfg)
    ple_ids = set(cfg.qwen4_args.ple_layer_ids)
    buf: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    plan = packing_plan(cfg)
    # part suffix -> (module, index, part count, role) for the packed modules
    packed_part: dict[str, tuple[str, int, int, str]] = {}
    for mod, parts in _PACKED_LINEAR.items():
        for i, part in enumerate(parts):
            packed_part[part] = (mod, i, len(parts), "qweight")
    for mod, (head, tail) in _PACKED_SPLIT.items():
        for i, part in enumerate(head):
            packed_part[part] = (mod, i, len(head), "packed.qweight")
        for i, part in enumerate(tail):
            packed_part[part] = (mod, i, len(tail), "dense")
    pbuf: dict[tuple[int, str, str], dict[int, torch.Tensor]] = {}
    top_key = {"model.embed_tokens": "model.embed_tokens.qweight", "lm_head": "lm_head.head.qweight"}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name in _TOP_LEVEL:
            mod = _TOP_LEVEL[name].removesuffix(".weight")
            if mod in plan:
                yield top_key.get(mod, f"{mod}.qweight"), t.packed()
            else:
                yield _TOP_LEVEL[name], _to_bf16(t)
            continue
        if name in _TOP_LEVEL_PLUS_ONE:
            yield _TOP_LEVEL_PLUS_ONE[name], _to_f32(t) - 1
            continue
        if name == PLE_TABLE_TENSOR:
            continue
        if not name.startswith("blk."):
            raise ValueError(f"unmapped qwen4exp GGUF tensor: {name}")
        _, layer_s, suffix = name.split(".", 2)
        layer = int(layer_s)
        if suffix in _EXPERT_SUFFIXES:
            continue
        base = f"model.layers.{layer}"

        hit = packed_part.get(suffix)
        if hit is not None and f"{base}.{hit[0]}" in plan:
            mod, idx, count, role = hit
            piece = _transform(suffix, t, vh) if role == "dense" else _packed_rows(suffix, t, vh)
            slots = pbuf.setdefault((layer, mod, role), {})
            slots[idx] = piece
            if len(slots) == count:
                del pbuf[(layer, mod, role)]
                rows = [slots[i] for i in range(count)]
                yield f"{base}.{mod}.{role}", rows[0] if count == 1 else torch.cat(rows, dim=0)
            continue

        if suffix in _PLUS_ONE:
            yield f"{base}.{_PLUS_ONE[suffix]}", _to_f32(t) - 1
        elif suffix in _PLAIN:
            yield f"{base}.{_PLAIN[suffix]}", _to_bf16(t)
        elif suffix == "ssm_out.weight":
            yield f"{base}.linear_attn.out_proj.weight", vh.untile(_to_bf16(t), 1, vh.value_dim)
        elif suffix == "ssm_a":
            # conversion: ssm_a = -exp(untiled^-1(A_log)); f32 end to end, the log is exact enough
            yield f"{base}.linear_attn.A_log", vh.untile_vec(torch.log(-_to_f32(t)))
        elif suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", vh.untile_vec(_to_f32(t))
        elif suffix == "ssm_conv1d.weight":
            w = _to_bf16(t)  # [channels, kernel]: the conversion squeezed HF's [C, 1, K]
            qk = 2 * vh.k_heads * vh.key_dim
            w = torch.cat([w[:qk], vh.untile(w[qk:], 0, vh.value_dim)], dim=0)
            yield f"{base}.linear_attn.conv1d.weight", w.unsqueeze(1)
        elif suffix == "ple_conv1d.weight":
            yield f"{base}.ple.conv1d.weight", _to_bf16(t).unsqueeze(1)
        elif suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
        elif suffix in _PART_OF:
            fused, idx = _PART_OF[suffix]
            slots = buf.setdefault((layer, fused), {})
            slots[idx] = _transform(suffix, t, vh)
            parts = _FUSED[fused]
            if len(slots) == len(parts):
                del buf[(layer, fused)]
                rows = [slots[i] for i in range(len(parts))]
                pad_to = next((p for sfx, p in _PAD_ROWS_TO.items() if fused.endswith(sfx)), 0)
                pad = (-sum(r.shape[0] for r in rows)) % pad_to if pad_to else 0
                if pad:
                    rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype))
                yield f"{base}.{fused}", torch.cat(rows, dim=0)
        else:
            raise ValueError(f"unmapped qwen4exp GGUF tensor: {name} ({GGML_NAME.get(t.ggml_type, t.ggml_type)})")

    assert not buf, f"incomplete fusions: {sorted(f'{layer}:{fused}' for layer, fused in buf)}"
    assert not pbuf, f"incomplete packed fusions: {sorted(pbuf)}"

    # The PLE hash constants ride the KV section (int64-exact) instead of tensors.
    meta = load_gguf_metadata(model_path)
    for lid in sorted(ple_ids):
        prefix = f"model.layers.{lid}.ple.ple_embedding"
        for key, attr in (
            ("ple.layer_multipliers", "layer_multipliers"),
            ("ple.head_vocab_sizes", "ngram_heads_vocab_sizes"),
            ("ple.head_offsets", "ngram_heads_offsets"),
        ):
            yield f"{prefix}.{attr}", torch.tensor([int(x) for x in meta[f"{_ARCH}.{key}"]], dtype=torch.int64)


__all__ = [
    "GGUFLMHead",
    "PLE_TABLE_TENSOR",
    "PackedSplitLinear",
    "convert_qwen4exp_to_gguf",
    "packing_plan",
    "dummy_q4_0_expert_sources",
    "iter_gguf_weights",
    "load_q4_0_expert_sources",
    "parse_gguf_config",
]
