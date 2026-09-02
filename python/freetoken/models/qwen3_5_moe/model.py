from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE
from .mtp import Qwen3_5MTP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        # MTP / nextn speculative head (opt-in via the engine's --mtp). Its attention block is
        # the appended full-attention layer (mtp_layer_id == num_layers); the KV pool carries
        # one extra slab for it (see ModelConfig.kv_cache_group_specs). When off (default) the
        # model is exactly the base model it was before -- no extra weights, no extra KV.
        if config.has_mtp:
            self.mtp = Qwen3_5MTP(config, config.mtp_layer_id)
        # Speculative-draft hook: after each decode forward the engine can read the post-norm
        # hidden of the just-processed token here (the head's prev_hidden input). Never part of
        # the state dict (leading underscore) and unused when MTP is off.
        self._last_hidden = None
        super().__init__()

    def _hidden(self) -> torch.Tensor:
        """Post-final-norm hidden states for the active batch ([N, hidden_size]), i.e. the
        input to ``lm_head`` -- also the MTP head's ``prev_hidden`` per position."""
        return self.model.forward(get_global_ctx().batch.input_ids)

    def forward(self) -> torch.Tensor:
        hidden = self._hidden()
        # Drafting hook: stash the post-final-norm hidden of the active batch (used by the MTP
        # speculative loop to seed the head; plain decode is unaffected).
        self._last_hidden = hidden
        return self.lm_head.forward(hidden)

    def forward_hidden(self) -> torch.Tensor:
        """Base-model hidden states without the lm_head (used by the speculative loop to seed
        the MTP draft; requires the active batch, like :meth:`forward`)."""
        return self._hidden()

    def forward_mtp(
        self, prev_hidden: torch.Tensor, next_ids: torch.Tensor
    ) -> torch.Tensor:
        """Draft logits from the MTP head: for each row of ``prev_hidden`` (a base hidden at
        position t) and ``next_ids`` (the token at t+1), returns logits over the token at t+2.
        Requires ``has_mtp`` (the engine only calls this on the --mtp path)."""
        assert self.mtp is not None, "forward_mtp requires the MTP head (serve with --mtp)"
        return self.mtp.forward(
            prev_hidden, next_ids, self.model.embed_tokens, self.lm_head
        )


__all__ = ["Qwen3_5MoEForCausalLM"]
