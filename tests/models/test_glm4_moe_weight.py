"""GLM-4 MoE checkpoints in either NVFP4 export.

NVIDIA's ModelOpt recipe leaves attention in bf16 and stores the dequant-side global scale;
llm-compressor's (GLM-4.5-Air REAP NVFP4) quantizes attention too and stores the quant-side
global. Both have to land as the same buffers: a global read the wrong way round is a model
that loads cleanly and answers garbage.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import torch

from freetoken.models.glm4_moe import weight as glm_weight

# the quantization_config the REAP NVFP4 export ships with
_CT_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4, "type": "float", "symmetric": True, "group_size": 16,
            "strategy": "tensor_group", "scale_dtype": "torch.float8_e4m3fn", "dynamic": False,
        },
        "input_activations": None,
        "format": "nvfp4-pack-quantized",
    }},
    "ignore": ["lm_head", "re:.*\\.mlp\\.gate$", "re:model\\.layers\\.46\\..*"],
}


class _Reader:
    def __init__(self, tensors):
        self._tensors = tensors

    def has(self, name):
        return name in self._tensors

    def get(self, name):
        return self._tensors[name]


def _nvfp4(rows=3, cols=32):
    packed = torch.arange(rows * cols // 2).remainder(256).to(torch.uint8).reshape(rows, cols // 2)
    scale = torch.ones(rows, cols // 16).to(torch.float8_e4m3fn)
    return packed, scale


def test_llm_compressor_global_is_inverted_to_the_dequant_side():
    packed, scale = _nvfp4()
    reader = _Reader({
        "p.weight_packed": packed, "p.weight_scale": scale,
        "p.weight_global_scale": torch.tensor([4.0]),
    })

    out = dict(glm_weight._iter_nvfp4_resident(reader, "p", "m"))

    assert out.keys() == {"m.weight", "m.weight_scale", "m.weight_global"}
    assert out["m.weight"] is packed and out["m.weight_scale"] is scale
    assert torch.equal(out["m.weight_global"], torch.full((3,), 0.25, dtype=torch.float16))


def test_llm_compressor_activation_global_is_inverted_too():
    packed, scale = _nvfp4()
    reader = _Reader({
        "p.weight_packed": packed, "p.weight_scale": scale,
        "p.weight_global_scale": torch.tensor([4.0]), "p.input_global_scale": torch.tensor([8.0]),
    })

    out = dict(glm_weight._iter_nvfp4_resident(reader, "p", "m"))

    assert out["m.input_scale"].item() == 0.125


def test_modelopt_global_is_read_as_stored():
    packed, scale = _nvfp4()
    reader = _Reader({
        "p.weight": packed, "p.weight_scale": scale,
        "p.weight_scale_2": torch.tensor(0.25), "p.input_scale": torch.tensor(2.0),
    })

    out = dict(glm_weight._iter_nvfp4_resident(reader, "p", "m"))

    assert torch.equal(out["m.weight_global"], torch.full((3,), 0.25, dtype=torch.float16))
    assert out["m.input_scale"].item() == 2.0


def test_either_naming_counts_as_nvfp4_and_bf16_does_not():
    assert glm_weight._is_nvfp4(_Reader({"a.q_proj.weight_packed": 0}), "a.q_proj")
    assert glm_weight._is_nvfp4(_Reader({"a.q_proj.weight_scale_2": 0}), "a.q_proj")
    assert not glm_weight._is_nvfp4(_Reader({"a.q_proj.weight": 0}), "a.q_proj")


def test_expert_reader_follows_the_export_and_skips_the_mtp_layer(monkeypatch):
    monkeypatch.setattr(
        glm_weight, "cached_load_hf_config", lambda path: SimpleNamespace(quantization_config=_CT_QUANT)
    )
    spec = glm_weight.nvfp4_expert_spec("reap", None)

    assert spec.global_reciprocal
    match = spec.key_pattern.match("model.layers.1.mlp.experts.95.down_proj.weight_packed")
    assert match and spec.kind_map[match["kind"]] == "weight"
    # the MTP layer's experts are bf16 and must not be taken for NVFP4 pieces
    assert spec.key_pattern.match("model.layers.46.mlp.experts.0.gate_proj.weight") is None
    config = SimpleNamespace(first_k_dense_replace=1, num_layers=46)
    assert spec.layer_to_bank(0, config) is None
    assert spec.layer_to_bank(1, config) == 0
    assert spec.layer_to_bank(46, config) is None

    monkeypatch.setattr(
        glm_weight, "cached_load_hf_config",
        lambda path: SimpleNamespace(quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4"}),
    )
    assert not glm_weight.nvfp4_expert_spec("nvidia", None).global_reciprocal


def _hf_config(quantization_config):
    return SimpleNamespace(
        architectures=["Glm4MoeForCausalLM"], model_type="glm4_moe",
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        vocab_size=128, hidden_act="silu", rms_norm_eps=1e-5, tie_word_embeddings=False,
        max_position_embeddings=4096, rope_theta=1_000_000.0, rope_scaling=None,
        partial_rotary_factor=0.5, n_routed_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True, use_qk_norm=False, first_k_dense_replace=1, n_shared_experts=1,
        routed_scaling_factor=1.0, n_group=1, topk_group=1, attention_bias=True,
        quantization_config=quantization_config,
    )


def test_attention_is_built_the_way_the_checkpoint_quantized_it():
    from freetoken.layers import LinearReplicated
    from freetoken.layers.quantization import QuantConfig
    from freetoken.models.glm4_moe.attention import Glm4MoeAttention
    from freetoken.models.glm4_moe.config import parse_config
    from freetoken.models.glm4_moe.df11_linear import LinearDF11

    hf = _hf_config(_CT_QUANT)
    config = dataclasses.replace(parse_config(hf), quant=QuantConfig.from_hf(hf))
    quantized = Glm4MoeAttention(config, 1, prefix="model.layers.1.self_attn")
    for proj in (quantized.q_proj, quantized.k_proj, quantized.v_proj, quantized.o_proj):
        assert isinstance(proj, LinearReplicated)

    plain = Glm4MoeAttention(dataclasses.replace(config, quant=None), 1, prefix="model.layers.1.self_attn")
    for proj in (plain.q_proj, plain.k_proj, plain.v_proj, plain.o_proj):
        assert isinstance(proj, LinearDF11)
