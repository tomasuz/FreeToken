import os
import time
from typing import TYPE_CHECKING, Tuple

import torch

from freetoken.utils.phase_timer import phase
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.fused import fused_topk
from freetoken.gguf_quant import GGUF_EXPERT_FORMATS
from freetoken.moe.offload_cache import OffloadMoeCache

# Compare what a worker produced against what came back: the same number if the handoff
# is sound, and the place the two diverge if it is not.
_TRACE_HANDOFF = os.environ.get("FREETOKEN_WORKER_TRACE") == "1"


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        # offload layers without a quant config stay on the format-tag banks (GGUF q4_0)
        self.quant_method = None
        if quant_config is not None or allocate_experts:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            if allocate_experts:
                self.quant_method.create_weights(self)

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self),
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._maybe_all_reduce(self._resident_gemm(hidden_states, topk_weights, topk_ids))



def _submit(executor, layer_id: int, hidden_states, topk_weights, ids):
    """Start work on an executor without waiting, whatever kind of executor it is.

    Every executor now takes the layer id: one CPU pool and one worker per device each
    serve all of them, so the identity of the layer is part of the request rather than of
    the executor. That is what lets the placement move a layer's work between devices from
    one step to the next.
    """
    return executor.decode_submit(layer_id, hidden_states, topk_weights, ids)


def _sync(executor, handle):
    return executor.decode_sync(handle)


def _assign_overflow(overflow, owner, shares: list[tuple[str, float]]) -> dict:
    """Divide the routes this device did not take among the other executors.

    Elementwise on purpose. The obvious way to split a set of routes is to find them and
    deal them out, but finding them means counting them, and counting on the device means
    asking the host how many -- a synchronisation, which a stream in the middle of a graph
    capture refuses outright. The two-way version this generalises never had the problem
    because it expressed its split as a mask; so does this one.

    ``owner`` labels every *position* with an executor, in the proportions the placement
    asked for, and is built without reference to which positions actually overflowed. A
    route goes to executor ``i`` where it overflowed and its position is labelled ``i`` --
    two elementwise ops, no counts, nothing the device has to tell the host.

    The cost of not counting is that the division is only proportional on average: which
    positions overflow is not correlated with the labelling, so over a step the shares come
    out right, and over a single route they may not. That is the correct trade for a
    quantity the placement is already smoothing over many steps.
    """
    return {
        name: overflow & (owner == index)
        for index, (name, _) in enumerate(shares)
    }


def _fill_owner_map(owner_host, shares: list[tuple[str, float]]) -> None:
    """Label each route position with the executor that will take it if it overflows.

    Written into pinned host memory that a captured graph copies from, so the labelling can
    change between replays without recapturing: the copy is a node, and a node re-reads its
    source every time it runs.

    Positions are dealt in contiguous runs rather than interleaved. Runs keep each
    executor's routes together, which matters for the one that reads them from a slot cache
    -- scattered routes touch more experts than clustered ones do.
    """
    weights = [max(0.0, w) for _, w in shares]
    total = float(sum(weights))
    n = owner_host.numel()
    if total <= 0.0:  # nothing measured yet: an equal division is the honest guess
        weights = [1.0] * len(shares)
        total = float(len(shares))
    flat = owner_host.view(-1)
    start = 0
    for index, weight in enumerate(weights):
        stop = n if index == len(weights) - 1 else start + int(round(n * weight / total))
        stop = min(max(stop, start), n)
        flat[start:stop] = index
        start = stop


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and hidden_states.shape[0] > 4:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and hidden_states.shape[0] > 4:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        with phase("mlp.topk"):
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_resident_layer(self.layer_id):
            return self._resident_expert_gemm(cache, hidden_states, topk_weights, topk_ids,
                                              is_prefill=False)
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        helpers = cache.split_helpers(self.layer_id)
        if helpers:
            return self._decode_split(
                cache, helpers, hidden_states, topk_weights, topk_ids
            )
        if cache.cpu_assist is not None:
            return self._decode_cpu_assist(cache, hidden_states, topk_weights, topk_ids)
        with phase("moe.ensure"):
            cache.ensure_experts(self.layer_id, topk_ids)
            if cache.collect_miss_hist:
                cache.record_miss_hist(self.layer_id)
        with phase("moe.fetch"):
            cache.copy_missing()
            cache.prefetch_join(self.layer_id)
        with phase("moe.gemm"):
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(self.layer_id),
                is_prefill=False,
            )

    def _decode_cpu_assist(self, cache, hidden_states, topk_weights, topk_ids):
        """Resident experts on the GPU, the rest on the CPU at the same time; nothing waits
        on PCIe. The CPU computes a missing expert from host RAM sooner than the link brings
        it over (Ryzen 7 5700G, Qwen3.8 UD-Q3_K_XL: ~100 us a route in the pool vs ~160 us
        to copy a 2 MiB expert), beside the GPU's GEMM instead of before it. The misses are
        still admitted to the slot cache, copied in the background for later steps.

        FREETOKEN_GGUF_CPU_ASSIST=1. On tm, decode 52.3 -> 44.7 ms/token; not for MTP
        verifies, whose extra misses make the CPU the longer side (T=3: 93 -> ~100 ms)."""
        executor = cache.cpu_assist
        with phase("moe.plan"):
            on_gpu = cache.assist_split(self.layer_id, topk_ids)
            cpu_ids = torch.where(on_gpu, topk_ids.new_full((), -1), topk_ids).contiguous()
            cpu_w = torch.where(on_gpu, topk_weights.new_zeros(()), topk_weights).contiguous()
        with phase("moe.submit.cpu"):
            pending = _submit(executor, self.layer_id, hidden_states, cpu_w, cpu_ids)
        with phase("moe.admit"):
            slots = cache.assist_plan(self.layer_id, topk_ids)
        with phase("moe.gemm"):
            gpu_slots = torch.where(on_gpu, slots, slots.new_zeros(())).to(topk_ids.dtype)
            gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
            out = self._expert_gemm(
                cache,
                hidden_states,
                gpu_w,
                gpu_slots,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(self.layer_id),
                is_prefill=False,
            )
        with phase("moe.join.cpu"):
            part = _sync(executor, pending)
        return out + part

    def _decode_split(
        self,
        cache: OffloadMoeCache,
        helpers: dict,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode this layer with its misses divided among every executor that can take them.

        This device always participates: it computes the experts it already holds, which
        cost nothing to reach, plus the share of the misses it is told to fetch. Everything
        it does not fetch goes to the other executors -- the CPU, a worker on another
        accelerator, or both -- in proportion to how fast each has actually been getting
        through experts. Rates come from :class:`~freetoken.moe.placement.RateTracker`,
        which learns them from these same steps, so a device that is slower than expected
        loses its share within a few tokens rather than at the next restart.

        Every executor is submitted before any is waited on. Waiting for each in turn would
        cost the sum of their times, and the split was computed to cost the longest.

        Each route is computed exactly once: the ids handed to an executor are ``-1``
        wherever a different one owns that route, and the partials are summed.
        """
        counts = cache.placement_counts
        capturing_now = torch.cuda.is_current_stream_capturing()
        with phase("moe.plan"):
            raw = topk_ids.clone()  # raw ids, before the kernel rewrites them to slots
            names = list(helpers)
            if counts:
                # The device reads a plan per miss count; an eager step writes this layer's.
                # Under capture the host rows stand as they are and the replay reads them.
                if not capturing_now:
                    cache.ensure_count_tables(topk_ids.numel(), len(names))
                    cache.plan_counts_for(self.layer_id, names)
            else:
                shares = cache.split_shares(self.layer_id, names)

        # The kernel fetches this fraction of the misses; the rest overflow to the helpers.
        # Under capture this device fetches every miss, and the helpers are left with
        # nothing. Not a preference: the capped fetch is not capture-safe. Its cap reaches
        # the kernel as a host-computed scalar, which a capture bakes into the node, while
        # the routing it caps is recomputed on every replay -- and the result is wrong
        # answers, reproducibly (correct with the cap lifted, garbage with it in place, on
        # the same graph and the same worker). A helper that idles costs a doorbell; a
        # helper fed by a frozen cap costs the answer.
        # Under capture the fraction written here would describe the tracing pass and then
        # stand for every replay, so capture leaves the vector alone: what a replay reads is
        # what OffloadMoeCache.refresh_placement wrote for that step, off the stream. An
        # eager step writes its own, and the two paths then agree on where the split lives.
        if not counts and not capturing_now:
            cache.set_fetch_fraction(self.layer_id, float(shares.get("gpu", 1.0)))
        with phase("moe.ensure"):
            cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # -> slot, or -1
            if cache.collect_stats:
                cache.record_decode_stats_hybrid(self.layer_id)
            on_gpu = topk_ids >= 0

            if counts:
                assignment = cache.assign_overflow_counts(self.layer_id, ~on_gpu, names)
            else:
                share_list = [(n, shares.get(n, 0.0)) for n in names]
                owner = cache.owner_map(self.layer_id, topk_ids.shape, share_list)
                assignment = _assign_overflow(~on_gpu, owner, share_list)
        pending, started = {}, {}
        for name, mask in assignment.items():
            # No "is this mask empty" test: answering it needs the device to tell the host
            # something, which a capture forbids. An executor handed a step with nothing in
            # it computes zeros for it, which the sum below adds harmlessly.
            ids = torch.where(mask, raw, raw.new_full((), -1)).contiguous()
            # Zero the weights this executor does not own, exactly as the device path does
            # for the routes it does not own. Marking an id -1 says "not yours" only to a
            # consumer that checks; the weight is what actually decides whether a route
            # contributes, so a route left with its weight adds whatever the -1 happened to
            # select. The sum below is over partials that must not overlap.
            weights = torch.where(mask, topk_weights, topk_weights.new_zeros(())).contiguous()
            started[name] = time.perf_counter()
            with phase(f"moe.submit.{name}"):
                pending[name] = _submit(helpers[name], self.layer_id, hidden_states,
                                        weights, ids)

        # Time the fetch alone. What the split needs from this device is the cost of one
        # *more* miss, and that is a transfer -- averaging it with the hits, which cost no
        # transfer at all, would report a device several times faster than its link and
        # hand it work the link cannot carry. The GEMM that follows is work this device
        # owes whatever the split decides, so it is charged as time already committed
        # rather than as part of the price of a miss.
        with phase("moe.fetch"):
            fetch_start = cache.record_event()
            cache.copy_missing()
            fetch_end = cache.record_event()

        with phase("moe.gemm"):
            gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
            gpu_w = torch.where(
                on_gpu, topk_weights, topk_weights.new_zeros(())
            ).contiguous()
            out = self._expert_gemm(
                cache,
                hidden_states,
                gpu_w,
                gpu_slots,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(self.layer_id),
                is_prefill=False,
            )
        cache.note_step_timing(fetch_start, fetch_end, cache.record_event())

        capturing = capturing_now
        for name, handle in pending.items():
            with phase(f"moe.join.{name}"):
                part = _sync(helpers[name], handle)
            if _TRACE_HANDOFF and not capturing:
                print(f"engine: layer {self.layer_id} <- {name} "
                      f"|y|={float(part.float().abs().sum()):.4f} "
                      f"routes={int(assignment[name].sum())}", flush=True)
            out = out + part
            if capturing:
                # A capture traces this once and replays the nodes; the Python around them
                # never runs again. A sample taken here would therefore describe the
                # tracing pass and then stand forever, which is worse than no sample --
                # and reading the count is a device-to-host read, which ends the capture
                # outright. The rates learned from the eager steps before capture stand.
                continue
            if getattr(helpers[name], "self_timed", False):
                # This executor measures itself, on its own stream, and does so every step
                # including the ones a graph replays. The clock below cannot: the span it
                # reads starts at submit and ends at join, so it contains this device's
                # fetch and GEMM as well -- an executor that overlaps perfectly is charged
                # the whole makespan and reads back an order of magnitude slow.
                continue
            # Count the routes this executor actually received. Reading a device tensor
            # costs a synchronisation, which is why this cannot stand under capture -- but
            # the alternative tried first, charging it the share it was given, measures a
            # different quantity from the one the main device is measured in, and the two
            # are then not comparable. That put a 1400 GB/s reading on the slower device
            # and sent it 99% of the work.
            cache.rate_tracker.observe(
                name, int(assignment[name].sum()), time.perf_counter() - started[name],
                cache.bytes_per_expert,
            )
        return out

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id, so the routing ids
        pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_resident_layer(self.layer_id):
            if self.layer_id == 0:
                # begin_prefill normally rides on _wait_prefill_overlap's layer-0 call; a
                # layer 0 that skips the movement path never gets there, so open it here.
                cache.begin_prefill()
            return self._resident_expert_gemm(cache, hidden_states, topk_weights, topk_ids,
                                              is_prefill=True)
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        if cache.quant_format == "gguf":
            return self._prefill_by_ensure(cache, hidden_states, topk_weights, topk_ids)
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _prefill_by_ensure(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill through the slot cache: only the experts this chunk routes to, and of
        those only the ones not already cached, cross PCIe.

        Materializing the layer instead copies every expert of it, cached or not: on
        Qwen3.8 GGUF that is 26 GiB per prefill however short the prompt (~2 s at PCIe 3.0
        x16), where a short prompt needs a fraction of the layer and much of that is still
        in the cache from decode. Not graph-captured (the id set is data-dependent), which
        prefill never is.
        """
        uniq = torch.unique(topk_ids)  # <= num_experts distinct ids, so the region holds them
        slots = uniq.to(torch.int32, copy=True)  # a copy: ensure_experts rewrites it in place
        cache.ensure_experts(self.layer_id, slots)
        cache.copy_missing()
        lut = torch.empty((self.num_experts,), dtype=slots.dtype, device=slots.device)
        lut[uniq.long()] = slots
        slot_ids = lut[topk_ids.long()].to(topk_ids.dtype)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            slot_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )

    def _resident_expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        is_prefill: bool,
    ) -> torch.Tensor:
        """Expert compute for a VRAM-resident layer: no movement at all.

        The layer's banks are already ``[num_experts, ...]`` on the device, so this is the
        materialized-prefill shape in both directions -- position == expert id, hence raw
        ``topk_ids`` (no slot rewrite) and ``alphas_for_layer``. Skipping
        ``ensure_experts``/``copy_missing`` is the whole point: nothing crosses PCIe and no
        slot is spent, which is why a resident layer needs no host bank to exist."""
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.resident_views(self.layer_id),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=is_prefill,
        )

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows.
    # GGUF q4_0 experts still dispatch on the cache's format tag until they get a method.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        if self.quant_method is not None:
            from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

            view = ExpertView(
                {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
                slots=None if n is not None else topk_ids, n=n, alphas=alphas,
            )
            return self.quant_method.apply(
                hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
            )
        fmt = cache.quant_format
        if fmt == "gguf":
            # Native GGUF experts whose ggml type varies by layer (and by projection): slot
            # rows are the widest layer's, so read them at this layer's width and types.
            # Resident banks are already this layer's own [E, rows, row_bytes].
            from freetoken.moe.fused_q4_0 import fused_experts_gguf

            gate_up, down = views
            if gate_up.dim() != 3:  # slot regions, not a resident layer's own banks
                gate_up, down = cache.layer_bank_views(self.layer_id, n)
            gate_up_type, down_type = cache.gguf_layer_types[self.layer_id]
            return fused_experts_gguf(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation,
                gate_up_type, down_type=down_type, num_experts=self.num_experts,
            )
        if fmt in GGUF_EXPERT_FORMATS:
            # Native GGUF experts (any ggml quant the borrowed kernels dispatch):
            # dequant-in-kernel grouped GEMV (MMVQ) over the streamed packed banks;
            # topk_ids already index the cache slots / layer.
            from freetoken.moe.fused_q4_0 import fused_experts_gguf

            gate_up, down = views
            return fused_experts_gguf(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation,
                GGUF_EXPERT_FORMATS[fmt],
            )
        raise AssertionError(f"offload experts without a quant method only serve native GGUF banks, got {fmt!r}")


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
