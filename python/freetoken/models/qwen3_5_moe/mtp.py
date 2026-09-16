"""Qwen3.5-MoE Multi-Token-Prediction (MTP / "nextn") head.

The base model predicts token ``t+1`` from the hidden state at ``t``.  The MTP head
predicts one token further ahead (``t+2``) from the base model's hidden state at ``t``
and the embedding of the token at ``t+1``:

    e = pre_fc_norm_embedding(embed(token_{t+1}))   # Gemma (1+w) RMSNorm
    h = pre_fc_norm_hidden(base_hidden_t)           # Gemma (1+w) RMSNorm
    x = fc(concat([e, h]))                          # Linear 2H -> H  (mtp.fc)
    x = mtp_layer(x)                                # one full-attn + MoE block (mtp.layers.0)
    x = norm(x)                                     # Gemma (1+w) RMSNorm   (mtp.norm)
    logits = lm_head(x)                             # SHARED with the base model

The module's state-dict keys match the checkpoint's ``mtp.*`` tensors 1:1 (19 tensors;
the routed experts are packed to the engine's ``experts.gate_up_proj``/``experts.down_proj``
storage by the weight loader, exactly like the base model's MoE layers).  Embedding and
lm_head are shared with the base model and are passed into :meth:`forward` by the engine.

Speculative use: run the head once per (base) decode step with ``prev_hidden`` =
the base model's final (pre-lm_head) hidden at the just-accepted position and
``next_ids`` = that position's token; the returned logits draft the token AFTER
``next_ids`` (t+2).  The inner self-attention is a plain causal full-attention block
(the ``+1`` layer appended after the base model's layers), so it keeps its own KV slot
and needs no GDN recurrent state to roll back on a rejected draft -- see the engine's
``--mtp`` speculative loop (freetoken/engine) for the propose/verify/commit wiring.

This is an opt-in extension: nothing runs unless a model is served with ``--mtp``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList

from .attention import Qwen3_5Attention
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

__all__ = ["Qwen3_5MTP", "Qwen3_5MTPLayer"]


class Qwen3_5MTPLayer(BaseOP):
    """One MTP transformer block.

    Structurally identical to a full-attention :class:`Qwen3_5DecoderLayer` (the nextn
    layer is always full attention, never GDN):

        x = x + self_attn(input_layernorm(x))
        x = x + mlp(post_attention_layernorm(x))

    ``layer_id`` is the block's layer id in the KV pool -- the appended full-attention
    layer right after the base model's layers (``num_layers``), so the pool must be
    sized for ``num_full_attention_layers + 1`` when ``--mtp`` is on.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        # The head is plain bf16 in the checkpoint -- every ``mtp.*`` tensor arrives as a
        # bare ``.weight``, with none of the scale tensors the base model's quantized
        # weights carry -- so the whole block is built from a config copy with the
        # quantization cleared. Building it quantized fails twice over: kernel selection
        # picks a table these weights cannot feed, and the engine, which casts each loaded
        # MTP tensor to its parameter's dtype, would cast bf16 values to a packed integer
        # dtype element by element.
        #
        # The copy also moves the MoE to the fused strategy. The head is the speculative
        # draft's hot path, so its experts stay RESIDENT on GPU instead of going to the
        # offload host banks, whatever the engine's --moe-strategy is; their packed
        # tensors ride the dense state dict.
        head_config = replace(
            config,
            moe_strategy="fused",
            expert_quant="none",
            moe_weight_format=None,
            attn_quant="none",
            dense_quant="none",
            quant=None,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3_5Attention(head_config, layer_id)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.moe_enabled:
            self.mlp = Qwen3_5MoE(head_config, layer_id)
        else:
            self.mlp = Qwen3_5DenseMLP(head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        return residual + x


class Qwen3_5MTP(BaseOP):
    """The nextn / MTP head.  State-dict keys match the checkpoint's ``mtp.*`` tensors:
    ``pre_fc_norm_embedding``, ``pre_fc_norm_hidden``, ``fc``, ``layers.0.*``, ``norm``.
    Embedding and lm_head are SHARED with the base model (passed into :meth:`forward`)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        h = config.hidden_size
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=config.rms_norm_eps)
        # mtp.fc.weight is [H, 2H]: projects concat(normed_embed, normed_hidden) -> H.
        self.fc = LinearReplicated(2 * h, h, has_bias=False)
        # OPList so the state-dict key is ``layers.0.*`` exactly as in the checkpoint.
        self.layers = OPList([Qwen3_5MTPLayer(config, layer_id)])
        self.norm = GemmaRMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        next_ids: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> torch.Tensor:
        """Draft logits for the token AFTER ``next_ids``.

        ``prev_hidden`` [T, H] is the base model's final hidden at each position (the
        value fed to ``lm_head`` -- post final-norm), ``next_ids`` [T] the corresponding
        next token ids; the returned logits [T, vocab] predict ``token_{t+2}`` per
        position.  The MTP self-attention reads positions/KV from the active forward
        batch (``get_global_ctx().batch``) like any other layer -- the caller must set
        the batch up so the MTP layer's KV slot holds the draft context (see the
        engine's speculative loop).
        """
        e = self.pre_fc_norm_embedding.forward(embed_tokens.forward(next_ids))
        h = self.pre_fc_norm_hidden.forward(prev_hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x = self.layers.op_list[0].forward(x)
        x = self.norm.forward(x)
        return lm_head.forward(x)
