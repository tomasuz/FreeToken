from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import QuantConfig, QuantKind
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


def _expert_quant(hf_config: Any, text: Any) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale block of block-fp8."""
    if not (getattr(text, "num_experts", 0) or 0):
        return "none", None
    # the engine reads this tag for its MoE strategy decisions; every module takes its own scheme from the QuantConfig when it is built
    scheme = QuantConfig.from_hf(hf_config).scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def _layer_types(text: Any) -> list[str]:
    layer_types = getattr(text, "layer_types", None)
    if layer_types is not None:
        return list(layer_types)
    # Fall back to full_attention_interval: every Nth layer (1-indexed) is full.
    interval = int(getattr(text, "full_attention_interval", 4))
    n = int(text.num_hidden_layers)
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def parse_config(hf_config: Any) -> ModelConfig:
    text = getattr(hf_config, "text_config", hf_config)

    head_dim = (
        getattr(text, "head_dim", None)
        or text.hidden_size // text.num_attention_heads
    )
    num_kv_heads = getattr(text, "num_key_value_heads", text.num_attention_heads)

    rope_params = (
        getattr(text, "rope_parameters", None)
        or getattr(text, "rope_scaling", None)
        or getattr(hf_config, "rope_scaling", None)
        or {}
    )
    rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
    partial = (
        rope_params.get("partial_rotary_factor")
        or getattr(text, "partial_rotary_factor", None)
        or 1.0
    )
    rotary_dim = int(head_dim * partial)

    # For text-only with the default rope type, partial NeoX rope needs no scaling dict
    # (the mRoPE params reduce to standard partial rope for text). Avoid carrying the
    # unhashable ``mrope_section`` list into get_rope's cache key.
    import os
    env_factor = os.environ.get("FREETOKEN_ROPE_FACTOR")
    env_type = os.environ.get("FREETOKEN_ROPE_TYPE")
    env_orig = os.environ.get("FREETOKEN_ROPE_ORIG_CTX")

    rope_type = (
        env_type
        if env_factor is not None and env_type
        else rope_params.get("rope_type") or rope_params.get("type") or "default"
    )
    if env_factor is not None and float(env_factor) > 1.0:
        factor = float(env_factor)
        orig_ctx = int(env_orig) if env_orig else text.max_position_embeddings
        rope_scaling = {
            "rope_type": str(rope_type).lower(),
            "factor": factor,
            "original_max_position_embeddings": orig_ctx,
        }
        if os.environ.get("FREETOKEN_ROPE_ATTN_FACTOR"):
            rope_scaling["attention_factor"] = float(os.environ["FREETOKEN_ROPE_ATTN_FACTOR"])
        if os.environ.get("FREETOKEN_ROPE_BETA_FAST"):
            rope_scaling["beta_fast"] = float(os.environ["FREETOKEN_ROPE_BETA_FAST"])
        if os.environ.get("FREETOKEN_ROPE_BETA_SLOW"):
            rope_scaling["beta_slow"] = float(os.environ["FREETOKEN_ROPE_BETA_SLOW"])
        max_position = int(orig_ctx * factor)
    elif rope_type in (None, "default"):
        rope_scaling = None
        max_position = text.max_position_embeddings
    else:
        rope_scaling = {k: v for k, v in rope_params.items() if not isinstance(v, (list, dict))}
        rope_scaling["rope_type"] = str(rope_type).lower()
        factor = float(rope_scaling.get("factor", 1.0))
        orig_pos = int(rope_scaling.get("original_max_position_embeddings", text.max_position_embeddings))
        max_position = int(orig_pos * factor) if factor > 1.0 else text.max_position_embeddings

    expert_quant, weight_block_size = _expert_quant(hf_config, text)

    # Dense variants (e.g. Qwen3.6-27B) report num_experts==0: route the decoder MLP through
    # the dense Qwen3_5DenseMLP instead of the MoE block.
    num_experts = getattr(text, "num_experts", 0) or 0
    moe_enabled = num_experts > 0

    layer_types = _layer_types(text)
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        scaling=rope_scaling,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=full_rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=text.linear_num_key_heads,
        num_value_heads=text.linear_num_value_heads,
        key_head_dim=text.linear_key_head_dim,
        value_head_dim=text.linear_value_head_dim,
        conv_kernel_dim=text.linear_conv_kernel_dim,
        output_gate="silu",
    )
    # Order groups by their first layer id for deterministic iteration.
    groups = tuple(
        sorted(
            (full_group, linear_group),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=text.num_hidden_layers,
        num_qo_heads=text.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        intermediate_size=getattr(text, "intermediate_size", 0),
        hidden_act=text.hidden_act,
        rms_norm_eps=text.rms_norm_eps,
        tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        shared_expert_intermediate_size=getattr(text, "shared_expert_intermediate_size", 0),
        norm_topk_prob=True,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type=getattr(hf_config, "model_type", "qwen3_5_moe"),
        architectures=getattr(hf_config, "architectures", ["Qwen3_5MoeForConditionalGeneration"]),
        vision_config=None,  # text-only milestone
        image_token_id=getattr(hf_config, "image_token_id", None),
        attention_groups=groups,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        # MTP / nextn speculative head geometry (kept even when --mtp is off; the engine
        # flips mtp_enabled on to build + serve the head).
        mtp_num_layers=int(getattr(text, "mtp_num_hidden_layers", 0) or 0),
        mtp_use_dedicated_embeddings=bool(
            getattr(text, "mtp_use_dedicated_embeddings", False)
        ),
    )


__all__ = ["parse_config"]
