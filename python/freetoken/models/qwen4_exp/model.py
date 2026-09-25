"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger, nvtx_annotate
from freetoken.utils.phase_timer import phase, step_done

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


# Experts guessed per token for the next layer's prefetch (0: off), and where the guess is
# taken: after this layer's attention ("post_attn") or before it ("pre_attn", a longer
# window for the copy, a staler residual for the guess).
_PREFETCH_K = int(os.getenv("FREETOKEN_MOE_PREFETCH_K", "0") or 0)
_PREFETCH_AT = os.getenv("FREETOKEN_MOE_PREFETCH_AT", "post_attn")
_PREFETCH_DEBUG = os.getenv("FREETOKEN_MOE_PREFETCH_DEBUG", "")  # "predict_only": cost of the guess alone


class _LayerRef:
    __slots__ = ("op",)

    def __init__(self, op) -> None:
        self.op = op


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMoE(config, layer_id, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple") if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    # the next decoder layer, behind a plain holder: the BaseOP tree and the MoE-layer walk
    # (which also enters lists and tuples) must not reach it twice
    _next: "_LayerRef | None" = None

    def prefetch_from(self, hidden: torch.Tensor) -> None:
        """Guess this layer's experts from an earlier residual and let the offload cache
        copy the misses in on a side stream while the previous layer still computes.

        The guess is this layer's own MLP mix and router applied to the earlier streams:
        the residual moves little across one block, so its top-k holds most of the real
        routing (from the pre-attention streams of the same layer, a top-12 held 90% of
        the real top-10 on Qwen3.8).
        """
        experts = getattr(self.mlp, "experts", None)
        cache = getattr(experts, "offload_cache", None)
        if cache is None or not cache.prefetch_supported(self._layer_id):
            return
        k = min(_PREFETCH_K, cache.num_experts, cache.prefetch_budget(self._layer_id) // hidden.shape[0])
        if k <= 0:
            return
        with phase("prefetch"):
            x, _ = self.mlp_hyper_connection.mix(hidden)
            ids = torch.topk(self.mlp.gate.forward(x), k, dim=-1).indices
            if _PREFETCH_DEBUG != "predict_only":
                cache.prefetch_begin(self._layer_id, ids)

    def _prefetch_next(self, hidden: torch.Tensor, batch: Batch) -> None:
        if self._next is not None and (batch.is_decode or getattr(batch, "mtp_verify", False)):
            self._next.op.prefetch_from(hidden)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            with phase("ple"):
                hidden = hidden + self.ple.forward(hidden, batch)
        if _PREFETCH_K and _PREFETCH_AT == "pre_attn":
            self._prefetch_next(hidden, batch)
        with phase("hc.mix"):
            block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            with phase("linear_attn"):
                block_output = self.linear_attn.forward(block_input)
        else:
            with phase("self_attn"):
                block_output = self.self_attn.forward(block_input, batch)
        with phase("hc.combine"):
            hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        if _PREFETCH_K and _PREFETCH_AT == "post_attn":
            self._prefetch_next(hidden, batch)
        with phase("hc.mix"):
            block_input, inject = self.mlp_hyper_connection.mix(hidden)
        with phase("mlp"):
            block_output = self.mlp.forward(block_input)
        with phase("hc.combine"):
            return self.mlp_hyper_connection.combine(hidden, block_output, inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        layers = self.layers.op_list
        for layer, nxt in zip(layers, layers[1:]):
            layer._next = _LayerRef(nxt)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        cache = getattr(get_global_ctx(), "moe_offload_cache", None)
        if cache is not None and getattr(cache, "cpu_assist", None) is not None:
            cache.assist_join()  # background admissions land before the next step
        if getattr(batch, "mtp_capture", False):
            # the MTP head drafts from the wide residual of every processed position
            self._mtp_residual = hidden
        with phase("hc.final"):
            out = self.hyper_connection_mixer.mix(hidden)[0]
        step_done()
        return out


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()
        if config.gguf_dense_types:
            # a GGUF checkpoint: its block-quantized projections run packed
            from .gguf import convert_qwen4exp_to_gguf

            convert_qwen4exp_to_gguf(self, config)

    # ----- MTP draft head (llama.cpp's separate MTP GGUF; see mtp.py) -----------------
    _mtp = None

    def init_mtp(self, mtp_path: str, device: torch.device) -> None:
        """Build the MTP head from ``mtp_path`` with its own expert cache. Kept out of the
        BaseOP tree: the engine's base expert cache and base state dict never see it."""
        import types

        from .mtp import (
            MTP_CACHE_SLOTS,
            Qwen4ExpMTPHead,
            build_mtp_cache,
            mtp_expert_banks,
            mtp_layer_and_experts,
            mtp_state_dict,
        )

        layer, experts, expert_types = mtp_layer_and_experts(mtp_path)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                head = Qwen4ExpMTPHead(self._config, experts)
        finally:
            torch.set_default_dtype(prev)
        want = head.state_dict()
        sd = {}
        for key, t in mtp_state_dict(mtp_path, layer).items():
            sd[key] = t.to(device=device, dtype=want[key].dtype)
        head.load_state_dict(sd)
        head.self_attn.alloc_ring(device, torch.bfloat16)
        cfg = self._config
        banks = mtp_expert_banks(mtp_path, layer, experts, cfg.moe_intermediate_size, cfg.hidden_size, expert_types)
        top_k = cfg.num_experts_per_tok
        cache = build_mtp_cache(device, experts, expert_types, banks, max(MTP_CACHE_SLOTS, 4 * top_k))
        head.mlp.experts.offload_cache = cache
        self._mtp = types.SimpleNamespace(head=head, cache=cache, banks=banks)
        mib = sum(v.numel() for v in cache.bank_caches.values()) / 2**20
        logger.info_rank0(f"MTP head from {mtp_path}: {experts} experts, {cache.lru_slots} cached ({mib:.0f} MiB)")

    @property
    def has_mtp_head(self) -> bool:
        return self._mtp is not None

    def mtp_forward(self, R_prev: torch.Tensor, next_ids: torch.Tensor, positions: torch.Tensor):
        """``(draft logits, head residual)`` for each (base residual at p, token at p+1)."""
        lm = self.lm_head
        head_linear = getattr(lm, "head", lm)  # GGUFLMHead wraps a GGUFLinear: all rows
        return self._mtp.head.forward(R_prev, next_ids, positions, self.model.embed_tokens, head_linear)

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        from freetoken.models.gguf.reader import is_gguf_path

        gguf = is_gguf_path(engine_config.model_path)
        if gguf and engine_config.ple_backend != "disk":
            # the GGUF table is one packed tensor (26.8 GiB IQ4_NL in unsloth's UD-Q3_K_XL);
            # pinning it would compete with the expert banks for host RAM, so read it by rows
            logger.info_rank0("qwen4exp GGUF: serving the PLE table from disk (--ple-backend disk)")
        if gguf or engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = engine_config.model_path if gguf else download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        h = self.model.forward(batch.input_ids, batch)
        with phase("lm_head"):
            if getattr(batch, "mtp_capture", False):
                # an MTP verify scores every row, not just each request's last
                head = getattr(self.lm_head, "head", None)
                assert head is not None, "MTP verify needs the GGUF LM head"
                return head.forward(h)
            return self.lm_head.forward(h)


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
