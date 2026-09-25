from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

from freetoken.gguf_quant import GGUF_EXPERT_FORMATS, row_bytes

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024
# "gguf" slot classes: a slot region per row width instead of one at the widest (0 = off).
_GGUF_SLOT_CLASSES = os.getenv("FREETOKEN_GGUF_SLOT_CLASSES", "1").strip().lower() not in {"0", "false", "no", "off"}

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native GGUF experts whose layers use different ggml types (unsloth's UD mixes): the
    # same two banks, but each layer's [E, rows, row_bytes] shape is its own. A slot holds
    # the widest layer's expert; each layer reads it through a view of its own width.
    "gguf": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
# Steps that may be in flight with their timing unread. Deeper costs pinned buffers and
# staler rates; shallower starts discarding measurements on a busy stream.
_TIMING_RING = 8

# Diagnostic: per layer-step histogram of how many experts had to be fetched (FREETOKEN_MISS_HIST=1).
# Wide enough for a batch of eight at top-8; a larger count lands in the last bin.
_MISS_HIST_BINS = 65

# Diagnostic: route every miss to the main device while leaving the split path in place.
_FORCE_GPU_ONLY = os.getenv("FREETOKEN_SPLIT_GPU_ONLY", "0") == "1"

_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# Native GGUF routed experts. Every ggml quant shares the same two-bank shape --
# gate_up [L*E, 2I, row_bytes(H)] and down [L*E, H, row_bytes(I)] -- and differs only
# in how many bytes a row of packed blocks takes, so register the whole family from
# one description instead of hand-writing a table row per quant. "q4_0" keeps its
# existing spelling and behaviour; the others are new.


def _gguf_bank_bytes(ggml_type: int):
    def sizer(H: int, I: int) -> int:
        return 2 * I * row_bytes(H, ggml_type) + H * row_bytes(I, ggml_type)

    return sizer


for _tag, _t in GGUF_EXPERT_FORMATS.items():
    _BANK_SCHEMAS.setdefault(_tag, ("gate_up", "down"))
    _BANK_BYTES_PER_EXPERT.setdefault(_tag, _gguf_bank_bytes(_t))

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # Layers a worker reads where they lie. Like the CPU layers, these need no device
        # address from this process -- the difference is only which executor computes them.
        self.inplace_layer_ids: frozenset = frozenset()
        # Per-layer (gate_up, down) ggml types of "gguf" banks; set by the engine.
        self.gguf_layer_types: list[tuple[int, int]] | None = None
        # "gguf" only: each bank's per-layer [rows, row_bytes] (the slot is the widest).
        self.bank_layer_shapes: dict[str, list[tuple[int, ...]]] = {}
        # "gguf" slot classes (_setup_gguf_parts): layers grouped by row width, each group
        # with its own slot region at its own width and its own slice of the LRU arrays.
        # None = one region at the widest layer's width (the pre-classes layout).
        self._parts: list[dict] | None = None
        self._layer_part: list[int] | None = None
        # LRU slots actually allocated; differs from cache_size (the engine's budget, in
        # widest-layer slots) only when slot classes turn one budget into more slots.
        self.lru_slots = self.cache_size
        if self.quant_format == "gguf" and (self.prefill_overlap or self.prefill_hit_d2d):
            # Both move whole layers or hit rows at one fixed width; materialize +
            # copy_missing are the paths that honor a width per layer.
            logger.info_rank0(
                "MoE gguf experts: layer row widths differ, so prefill overlap and the hit-D2D "
                "gather are off (prefill materializes each layer)"
            )
            self.prefill_overlap = False
            self.prefill_hit_d2d = False
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # GPU-RESIDENT layers: their experts live in VRAM only (no host copy at all), so
        # they never touch the slot cache. Set by set_resident_banks BEFORE
        # set_bank_sources, which needs the id set to skip them in the copy plan --
        # their host sources are released mmaps whose pages are gone.
        # resident_banks: {bank name: {layer_id: [num_experts, ...] device tensor}}.
        self.resident_layer_ids: frozenset = frozenset()
        self.resident_banks: dict[str, dict[int, torch.Tensor]] = {}
        # Layers served by a worker process on another device. Like the resident ones they
        # are excluded from every movement path -- the worker owns their weights, in shared
        # memory of its own -- but the compute lands elsewhere rather than here.
        self.worker_layer_ids: frozenset = frozenset()
        self.worker_executors: dict = {}
        # Other devices of this process, each serving whatever layers it was offered. A
        # worker is the same thing across a process boundary; these need none, so they
        # share the banks rather than a copy of them (see moe/device_executor.py).
        self.device_executors: list = []
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        # Diagnostic, off unless FREETOKEN_MISS_HIST=1: how many experts each layer-step had
        # to fetch. The mean hides what a placement decision acts on -- whether the misses
        # come a few at a time or many at once -- and a helper can only pay on the steps
        # with many. One launch per layer per step when on, captured like the rest.
        self.collect_miss_hist = os.getenv("FREETOKEN_MISS_HIST", "0") == "1"
        self.miss_hist = torch.zeros(
            (self.num_layers, _MISS_HIST_BINS), dtype=torch.int64, device=self.device
        )
        self._miss_hist_one = torch.ones((1,), dtype=torch.int64, device=self.device)
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # How fast each executor gets through experts, learned from the steps it runs. The
        # split between this device and any others reads it; see freetoken.moe.placement.
        # Seeded from the benchmark profile where the engine has one, else from the first
        # steps' own measurements.
        from freetoken.moe.placement import RateTracker

        self.rate_tracker = RateTracker()
        self._gpu_busy_seconds = 0.0
        self._hybrid_ensure_ran = False
        self._owner_maps: dict = {}
        # The capped fetch reaches the kernel through these rather than as a launch
        # argument. A launch argument is a host value, and a capture bakes it into the node
        # -- the split would then be frozen at whatever it was when the graph was recorded,
        # while the routing it caps goes on being recomputed every replay. A pinned host
        # vector plus a copy issued inside the captured region makes it a node input: the
        # replay re-reads the host bytes, so writing them between replays moves the split.
        seed = min(1 << 16, max(0, round(self.hybrid_fetch_fraction * (1 << 16))))
        self._fetch_frac_host = torch.full(
            (self.num_layers,), seed, dtype=torch.int32, device="cpu"
        ).pin_memory()
        self._fetch_frac_dev = torch.full(
            (self.num_layers,), seed, dtype=torch.int32, device=self.device
        )
        self._pending_timings: list = []
        # FREETOKEN_PLACEMENT=counts: instead of one fraction per layer, the device looks up
        # -- per layer and per miss count -- how many misses it fetches and how the rest
        # divide among the helpers (freetoken.moe.placement.plan_miss_counts). The costs
        # behind the plan are fitted from what each self-timed executor reports.
        from freetoken.moe.placement import CostTracker

        self.placement_counts = os.getenv("FREETOKEN_PLACEMENT", "shares") == "counts"
        self.cost_tracker = CostTracker()
        self._count_width = 0
        self._fetch_table_host: torch.Tensor | None = None
        self._fetch_table_dev: torch.Tensor | None = None
        self._helper_bounds_host: torch.Tensor | None = None
        self._helper_bounds_dev: torch.Tensor | None = None
        self._count_layers: dict[int, tuple[str, ...]] = {}
        self._count_plan_memo: dict[tuple[str, ...], tuple[tuple, list]] = {}
        self._count_rows_last: dict[tuple[str, ...], list] = {}
        self._fetch_calibrated = False
        self._fetched_staging: list = []
        self._timing_slot = 0
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to an executor that does not need one -- the CPU (``cpu_layer_ids``) or a worker reading the bank in place (``inplace_layer_ids``), either set BEFORE this call. The copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            addressless_ok = self.cpu_layer_ids | self.inplace_layer_ids
            if not unpinned <= addressless_ok:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - addressless_ok)} are in neither "
                    f"cpu_layer_ids nor inplace_layer_ids: a layer without a device address "
                    f"can only be computed by an executor that does not need one -- the CPU, "
                    f"or a worker reading the bank where it lies (set either before "
                    f"set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            if self.layout is not None:
                spec = self.layout[name]
                if tuple(head.shape[1:]) != tuple(spec.shape) or head.dtype != spec.dtype:
                    raise ValueError(
                        f"bank {name!r} rows are {tuple(head.shape[1:])} {head.dtype} but the expert kernel's layout "
                        f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                    )
            for layer_id, source in enumerate(per_layer):
                if self._skips_movement(layer_id):
                    continue  # released host bank: shape is still right, the pages are not there
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                if self.quant_format == "gguf":
                    # packed ggml rows; the row width is the layer's own
                    assert source.dtype == torch.uint8 and source.dim() == 3, (
                        name, layer_id, source.shape, source.dtype,
                    )
                else:
                    assert source.shape == head.shape and source.dtype == head.dtype, (
                        name, layer_id, source.shape, source.dtype,
                    )
            self.bank_sources[name] = list(per_layer)
            if self.quant_format == "gguf":
                self.bank_layer_shapes[name] = [tuple(t.shape[1:]) for t in per_layer]
            else:
                self.bank_caches[name] = self._alloc_slot_cache(name, self.cache_size)
        if self.quant_format == "gguf":
            self._alloc_gguf_slots(self.cache_size)
            if self.id_of_slot.numel() != self.lru_slots:
                # nothing has been cached yet: the LRU state just follows the slot count
                self.id_of_slot = torch.full((self.lru_slots,), -1, dtype=torch.int32, device=self.device)
                self.usage = torch.zeros((self.lru_slots,), dtype=torch.int64, device=self.device)
                plan_slots = max(self.num_experts, self.lru_slots)
                self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
                self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _alloc_slot_cache(self, name: str, cache_size: int) -> torch.Tensor:
        """One bank's GPU slot cache: ``[cache_size, *row shape]`` in the bank dtype ("gguf"
        banks go through :meth:`_alloc_gguf_slots` instead)."""
        head = self.bank_sources[name][0]
        return torch.empty((cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device)

    def _layer_widths(self, layer_id: int) -> tuple[int, ...]:
        return tuple(math.prod(self.bank_layer_shapes[n][layer_id]) for n in self.bank_schema)

    def _plan_gguf_parts(self, budget_slots: int) -> list[dict]:
        """Group the layers by row width and size each group's slot region.

        ``budget_slots`` is the engine's budget in widest-layer slots, i.e. that many times
        the widest expert in bytes. One region at the widest width wastes what every
        narrower layer leaves of a slot (unsloth's UD mixes: 43 of 48 Qwen3.8 layers use
        2.08 MiB of a 3.32 MiB slot), so the most common width gets a region of its own
        and the rest share one at their widest. Every layer is given the same number of
        slots; a region holds at least ``num_experts`` (prefill materializes a whole
        layer into it).
        """
        widths = [self._layer_widths(l) for l in range(self.num_layers)]
        widest = tuple(max(w[i] for w in widths) for i in range(len(self.bank_schema)))
        budget = budget_slots * sum(widest)
        common = max(set(widths), key=widths.count)
        groups = [[l for l in range(self.num_layers) if widths[l] == common]]
        rest = [l for l in range(self.num_layers) if widths[l] != common]
        if rest:
            groups.append(rest)
        if not _GGUF_SLOT_CLASSES or len(groups) == 1:
            groups = [list(range(self.num_layers))]
        parts = []
        for layers in groups:
            w = tuple(max(widths[l][i] for l in layers) for i in range(len(self.bank_schema)))
            parts.append({"layers": tuple(layers), "widths": w})
        # equal slots per layer, then lift any region under the num_experts floor and give
        # the others what is left
        per_layer_bytes = sum(len(p["layers"]) * sum(p["widths"]) for p in parts)
        per_layer = budget // per_layer_bytes
        floored = [p for p in parts if len(p["layers"]) * per_layer < self.num_experts]
        for p in floored:
            p["size"] = self.num_experts
        free = [p for p in parts if p not in floored]
        if free:
            left = budget - sum(p["size"] * sum(p["widths"]) for p in floored)
            per_layer = left // sum(len(p["layers"]) * sum(p["widths"]) for p in free)
            for p in free:
                p["size"] = max(self.num_experts, len(p["layers"]) * per_layer)
        base = 0
        for p in parts:
            p["base"] = base
            base += p["size"]
        return parts

    def _alloc_gguf_slots(self, budget_slots: int) -> None:
        """Allocate the "gguf" slot regions: one flat uint8 buffer per bank, a region per
        slot class inside it, each read at its class width (:meth:`layer_bank_views`)."""
        parts = self._plan_gguf_parts(budget_slots)
        self._parts = parts
        self._layer_part = [0] * self.num_layers
        for i, p in enumerate(parts):
            for l in p["layers"]:
                self._layer_part[l] = i
        self.lru_slots = sum(p["size"] for p in parts)
        for b, name in enumerate(self.bank_schema):
            total = sum(p["size"] * p["widths"][b] for p in parts)
            flat = torch.empty((total,), dtype=torch.uint8, device=self.device)
            self.bank_caches[name] = flat
            offset = 0
            for p in parts:
                p.setdefault("views", {})[name] = flat[offset : offset + p["size"] * p["widths"][b]].view(
                    p["size"], p["widths"][b]
                )
                offset += p["size"] * p["widths"][b]
        if len(parts) > 1:
            desc = ", ".join(
                f"{len(p['layers'])} layers x {p['size'] // len(p['layers'])} slots of "
                f"{sum(p['widths']) / 2**20:.2f} MiB" for p in parts
            )
            logger.info_rank0(
                f"MoE gguf slot classes: {desc} = {self.lru_slots} slots "
                f"(one widest-width region would hold {budget_slots})"
            )

    def lru_part(self, layer_id: int) -> tuple[int, int]:
        """``(base, size)`` of the slot region ``layer_id`` lives in (the whole cache
        without slot classes)."""
        if self._parts is None:
            return 0, self.lru_slots
        p = self._parts[self._layer_part[layer_id]]
        return p["base"], p["size"]

    def layer_bank_views(self, layer_id: int, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """:meth:`bank_views` as ``layer_id`` reads them. For "gguf" banks each slot of the
        layer's region is viewed as that layer's ``[rows, row_bytes]`` (rows packed at the
        slot's start, slots a region-width apart), indexed by region-local slot; every
        other format's views are already the layer's."""
        if self.quant_format != "gguf":
            return self.bank_views(n)
        part = self._parts[self._layer_part[layer_id]]
        out = []
        for name in self.bank_schema:
            flat = part["views"][name]
            if n is not None:
                flat = flat[:n]
            rows, row_bytes = self.bank_layer_shapes[name][layer_id]
            out.append(flat.as_strided((flat.size(0), rows, row_bytes), (flat.stride(0), row_bytes, 1)))
        return tuple(out)

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            return
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            feat = math.prod(cache.shape[1:]) * cache.element_size()
            if feat % 128:
                raise RuntimeError(
                    f"MoE bank {name!r} rows are {feat} bytes (not a multiple of 128): "
                    f"only the fused multi-bank copy can move them, but it is disabled"
                )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        # "gguf": per-layer [num_banks] row bytes, and the slot stride they land at
        self._copy_layer_feat_bytes: list[torch.Tensor] | None = None
        self._copy_layer_dst_ptrs: list[torch.Tensor] | None = None
        self._copy_layer_dst_stride: list[torch.Tensor] | None = None
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        mixed = self.quant_format == "gguf"
        layer_feats = [[] for _ in range(self.num_layers)]
        for b, (per_layer, cache) in enumerate(self.banks):
            if mixed:
                # a "gguf" slot is its region's width (its slot class); the flat bank buffer
                # holds the regions back to back
                for part in self._parts:
                    view = part["views"][self.bank_schema[b]]
                    if view.stride(0) % 16 or view.data_ptr() % 16:
                        raise RuntimeError(
                            f"gguf MoE slot rows are {view.stride(0)} B: the fused copy needs 16-byte multiples"
                        )
                feat = max(part["widths"][b] for part in self._parts)
            else:
                feat = cache[0].numel() * cache.element_size()
                if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                    return  # leave fused disabled; copy_missing uses the per-bank path
            if mixed:
                for layer_id, source in enumerate(per_layer):
                    lf = math.prod(source.shape[1:]) * source.element_size()
                    if lf % 16:
                        raise RuntimeError(f"gguf MoE layer {layer_id} rows are {lf} B, not a 16-byte multiple")
                    layer_feats[layer_id].append(lf)
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers or self._skips_movement(layer_id):
                    # unregistered layer: no device alias exists, and the row is never consumed
                    # (CPU decode; pageable prefill; or -- for a resident layer -- the host bank
                    # was uploaded and released, so its pages are gone and nothing may read it)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        if mixed:
            self._copy_layer_feat_bytes = [
                torch.tensor(f, dtype=torch.int64, device=self.device) for f in layer_feats
            ]
            # per layer: its region's base address and slot width, one entry per bank
            part_dst = [
                torch.tensor([part["views"][n].data_ptr() for n in self.bank_schema], dtype=torch.int64, device=self.device)
                for part in self._parts
            ]
            part_stride = [
                torch.tensor(list(part["widths"]), dtype=torch.int64, device=self.device) for part in self._parts
            ]
            self._copy_layer_dst_ptrs = [part_dst[self._layer_part[l]] for l in range(self.num_layers)]
            self._copy_layer_dst_stride = [part_stride[self._layer_part[l]] for l in range(self.num_layers)]
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.max_slots is not None and cache_size > self.max_slots:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        if self.quant_format == "gguf":
            self._alloc_gguf_slots(cache_size)
        else:
            self.lru_slots = cache_size
            for name in self.bank_schema:
                self.bank_caches[name] = self._alloc_slot_cache(name, cache_size)
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((self.lru_slots,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((self.lru_slots,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, self.lru_slots)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_inplace_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` is computed wholly by a worker reading the bank in place.

        Such a layer is not pinned here and so has no device address: this process can
        neither fetch its experts nor give it a slot, which is the point -- the bytes are
        already where the worker's device can read them.
        """
        return layer_id in self.inplace_layer_ids

    def set_resident_banks(
        self, resident: dict[str, dict[int, torch.Tensor]], layers: frozenset
    ) -> None:
        """Register the VRAM-only expert banks for ``layers`` (call BEFORE set_bank_sources).

        These layers are excluded from every movement path: no slot is ever assigned to
        them, ``copy_missing`` never reads their host sources (whose pages the loader
        released), and their MoE layers read ``resident_views`` directly with raw expert ids
        -- position == expert id, exactly like a materialized prefill layer."""
        if not layers:
            self.resident_layer_ids = frozenset()
            self.resident_banks = {}
            return
        assert set(resident) == set(self.bank_schema), (
            f"resident banks {sorted(resident)} do not match the "
            f"{self.quant_format!r} schema {self.bank_schema}"
        )
        for name, per_layer in resident.items():
            missing = sorted(layers - set(per_layer))
            assert not missing, f"resident bank {name!r} missing layers {missing}"
            for layer_id in layers:
                t = per_layer[layer_id]
                # A tensor's device always carries an index; self.device may be the bare
                # "cuda" the caller constructed the cache with, so compare index-tolerantly.
                assert t.device.type == self.device.type and (
                    self.device.index is None or t.device.index == self.device.index
                ), (name, layer_id, t.device, self.device)
                assert t.size(0) == self.num_experts, (name, layer_id, t.shape)
        overlap = layers & self.cpu_layer_ids
        assert not overlap, (
            f"layers {sorted(overlap)} are both GPU-resident and CPU-decode; "
            "--moe-resident-layers and --moe-cpu-layers must not overlap"
        )
        self.resident_layer_ids = frozenset(layers)
        self.resident_banks = {name: dict(per) for name, per in resident.items()}

    def is_resident_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s experts live in VRAM only (no host bank, no slot cache)."""
        return layer_id in self.resident_layer_ids

    def set_worker_executors(self, executors: dict) -> None:
        """Attach per-layer worker executors (call BEFORE set_bank_sources).

        Their layers stay in the copy plan. A worker used to own its layer outright, so
        this side had no reason to be able to move those bytes; now the layer is shared --
        the placement gives some of each step's misses to the worker and the rest to this
        device -- and this device can only take its share if the host bank is still
        reachable. It is: the bank was never released, only excluded, which is the "held
        twice" the worker wiring reports. Both consumers now read the copy that was already
        being paid for."""
        overlap = frozenset(executors) & self.cpu_layer_ids
        assert not overlap, (
            f"layers {sorted(overlap)} are assigned to both a worker and CPU decode"
        )
        # No resident check: a resident layer's bank was released, so it has no shared file
        # and a worker cannot list it. The catalogue drops those layers rather than this
        # having to forbid them.
        self.worker_executors = dict(executors)
        self.worker_layer_ids = frozenset(executors)

    def is_worker_layer(self, layer_id: int) -> bool:
        """Whether a worker on another device can compute ``layer_id``'s experts.

        "Can", not "does": the worker is one of the executors this layer's misses are
        divided among, and how many it gets is decided per step by :meth:`split_shares`.
        """
        return layer_id in self.worker_layer_ids

    # --- placing a step's misses across whatever executors this layer has ---------------

    @property
    def bytes_per_expert(self) -> int:
        """One expert's weight across every bank, measured from the banks themselves.

        Taken from the registered tensors rather than the format table because the table is
        keyed by the config-time tag and this only needs a number that is right for the
        banks actually loaded. The split is scale-invariant, so this matters for the
        reported rates rather than for the decision -- but a rate in the wrong units is a
        log line nobody can check against a bandwidth measurement.
        """
        if not self.banks:
            return 1
        total = 0
        for _, cache_tensor in self.banks:
            total += cache_tensor[0].numel() * cache_tensor.element_size()
        return max(1, total)

    def split_helpers(self, layer_id: int) -> dict:
        """Executors besides this device that can take some of ``layer_id``'s misses.

        Empty means there is nothing to divide and the caller should take the plain path;
        that is the ordinary single-device case and it must stay free of any of this.
        """
        helpers: dict = {}
        if self.decode_target == "hybrid" and self.cpu_executor is not None:
            helpers["cpu"] = self.cpu_executor
        worker = self.worker_executors.get(layer_id)
        if worker is not None and worker.serves(layer_id):
            helpers[f"worker{getattr(worker, 'device_index', '?')}"] = worker
        for executor in self.device_executors:
            if executor.serves(layer_id):
                helpers[f"gpu{executor.device_index}"] = executor
        return helpers

    def owner_map(self, layer_id: int, shape, shares: list) -> torch.Tensor:
        """Per-route executor labels for this layer, as a device tensor a graph can read.

        Kept as a pinned host buffer plus a device copy, and the copy is issued here on the
        stream. A captured graph records the copy as a node, so a later step can change the
        labelling by writing the host buffer and the replay picks it up -- the division
        stays adjustable without recapturing anything.

        The host buffer is rewritten only when the shares actually move, since rewriting it
        is Python over every route position and the shares are already smoothed.
        """
        from freetoken.layers.moe import _fill_owner_map

        key = (layer_id, tuple(shape))
        entry = self._owner_maps.get(key)
        if entry is None:
            host = torch.zeros(shape, dtype=torch.int32, device="cpu").pin_memory()
            device = torch.zeros(shape, dtype=torch.int32, device=self.device)
            entry = self._owner_maps[key] = [host, device, None]
        host, device, previous = entry
        current = tuple(round(w, 3) for _, w in shares)
        if current != previous:
            _fill_owner_map(host, shares)
            entry[2] = current
        device.copy_(host, non_blocking=True)
        return device

    def seed_fetch_fraction(self, gpu_share: float) -> None:
        """One fraction for every layer, for a cap that arrives before the first step.

        The `ft bench bw` profile measures the machine, not a layer, so it seeds all of
        them; the placement then moves each one on its own as the steps report in.
        """
        self.hybrid_fetch_fraction = gpu_share
        self._fetch_frac_host.fill_(min(1 << 16, max(0, round(gpu_share * (1 << 16)))))
        self._fetch_frac_dev.copy_(self._fetch_frac_host)

    def set_fetch_fraction(self, layer_id: int, gpu_share: float) -> None:
        """This layer's capped-fetch share, written where the next launch will read it."""
        self._fetch_frac_host[layer_id] = min(1 << 16, max(0, round(gpu_share * (1 << 16))))

    def refresh_placement(self) -> None:
        """Recompute every split this cache has a labelling for, and write it where a replay reads it.

        A captured decode runs no Python: :meth:`split_shares` is called while the graph is
        being recorded and never again, so without this the division a graph was captured
        with is the division it keeps -- and since capture is what makes the fetch cap
        unsafe, that division is "this device takes everything". Both halves of the labelling
        are already node inputs (the owner map's pinned host buffer, the fetch fraction
        vector), so a step only has to rewrite the host bytes; the replay picks them up.

        Called once per step from outside the graph. That keeps the work off the stream, but
        not free: the replay is not launched until this returns, so every microsecond here
        is a microsecond on every token. Layers with no helper are skipped -- there is
        nothing to divide, and the plain path must stay free of any of this.

        The division is computed once per distinct set of helpers, not once per layer. It
        depends on which executors take part and on the rates they share, and on nothing
        about the layer, so layers with the same helpers get the same answer. Recomputing
        it for each of forty layers was ~4 ms of host time in front of a ~25 ms step.
        """
        from freetoken.layers.moe import _fill_owner_map

        if self.placement_counts:
            self._refresh_count_placement()
            return
        if not self._owner_maps:
            return
        # helper names -> (labelled shares, their rounded signature, fetch fraction in Q16)
        splits: dict[tuple[str, ...], tuple[list, tuple, int]] = {}
        for (layer_id, _shape), entry in self._owner_maps.items():
            helpers = self.split_helpers(layer_id)
            if not helpers:
                continue
            key = tuple(helpers)
            split = splits.get(key)
            if split is None:
                shares = self.split_shares(layer_id, list(helpers))
                share_list = [(name, shares.get(name, 0.0)) for name in helpers]
                gpu = shares.get("gpu", 1.0)
                split = splits[key] = (
                    share_list,
                    tuple(round(w, 3) for _, w in share_list),
                    min(1 << 16, max(0, round(gpu * (1 << 16)))),
                )
            share_list, current, fraction = split
            if current != entry[2]:
                _fill_owner_map(entry[0], share_list)
                entry[2] = current
            self._fetch_frac_host[layer_id] = fraction

    # --- the device's per-miss-count lookup (FREETOKEN_PLACEMENT=counts) -----------------

    def ensure_count_tables(self, routes: int, helpers: int) -> None:
        """Size the lookup tables for ``routes`` routes a step and up to ``helpers`` helpers.

        Allocated on the first split -- an eager step before any capture, the only time an
        allocation a graph will read can be made. Until a plan is written every row says
        "this device fetches everything", which is always a correct answer.
        """
        width = int(routes) + 1
        helpers = max(1, int(helpers))
        if (self._fetch_table_host is not None and width <= self._count_width
                and helpers <= self._helper_bounds_host.shape[1]):
            return
        assert not torch.cuda.is_current_stream_capturing(), "count tables must exist before capture"
        width = max(width, self._count_width)
        pin = self.device.type == "cuda"
        table = torch.arange(width, dtype=torch.int32).expand(self.num_layers, width).contiguous()
        self._fetch_table_host = table.pin_memory() if pin else table
        self._fetch_table_dev = self._fetch_table_host.to(self.device)
        bounds = torch.zeros((self.num_layers, helpers, width), dtype=torch.int32)
        self._helper_bounds_host = bounds.pin_memory() if pin else bounds
        self._helper_bounds_dev = self._helper_bounds_host.to(self.device)
        self._count_width = width
        self._count_rows_last.clear()
        self.calibrate_fetch_cost()

    def calibrate_fetch_cost(self, sizes: tuple[int, ...] = (1, 2, 4, 8), repeats: int = 3) -> None:
        """Measure this device's fetch directly, once, before anything is captured.

        A decode driven by graph replays takes its only fetch samples in the eager steps that
        warm up and record the graphs, and in those the host is still compiling between the
        two stream markers: the stream idles inside the window, and one expert reads 9.4 ms
        where the link moves it in 0.7. Timed copies of real bank rows -- a few sizes, the best
        of a few tries each -- give the cost the way a decode step pays it, with nothing else
        in the window. From then on the warm-up samples are not used for the plan.
        """
        import time

        if self._fetch_calibrated or self.device.type != "cuda" or not self.banks:
            return
        layer = next(
            (l for l in range(self.num_layers)
             if l not in self._unpinned_layers and not self._skips_movement(l)),
            None,
        )
        if layer is None:
            return
        rows = min(max(sizes), self.num_experts)
        scratch = [
            torch.empty_like(per_layer[layer][:rows], device=self.device)
            for per_layer, _ in self.banks
        ]
        for count in sizes:
            if count > rows:
                continue
            best = None
            for _ in range(repeats):
                torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                for (per_layer, _), dst in zip(self.banks, scratch):
                    dst[:count].copy_(per_layer[layer][:count], non_blocking=True)
                torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - started
                best = elapsed if best is None else min(best, elapsed)
            self.cost_tracker.observe("gpu", 1, count, best)
            # Feed the shares path too. It reads rate_tracker, which is filled only by
            # note_step_timing -- and that lives inside the captured region, so under a
            # graph-driven decode it never runs again after the warm-up. Without this the
            # counts path got a measured fetch and the shares path kept the warm-up guess.
            self.rate_tracker.observe("gpu", count, best, self.bytes_per_expert)
        del scratch
        self._fetch_calibrated = True

    def _executor_costs(self, names: list[str]):
        """This device and each helper as fixed + per-expert seconds, from what they reported."""
        from freetoken.moe.placement import ExecutorCost

        self._drain_self_timed()
        rates = {r.name: r for r in self.rate_tracker.rates(["gpu", *names])}

        def per_expert(name: str) -> float:
            rate = rates.get(name)
            if rate is None or not rate.usable:
                return 1.0  # unmeasurable: a second per expert keeps it out of the plan
            return self.bytes_per_expert / rate.bytes_per_second

        busy = getattr(self, "_gpu_busy_seconds", 0.0)
        fitted_gpu = self.cost_tracker.cost("gpu")
        if fitted_gpu is None:
            main = ExecutorCost("gpu", per_expert("gpu"), busy)
        else:  # the copy is launched every layer, so its fixed part is paid regardless
            main = ExecutorCost("gpu", fitted_gpu[1], busy + fitted_gpu[0])
        helpers = []
        for name in names:
            fitted = self.cost_tracker.cost(name)
            if fitted is None:
                helpers.append(ExecutorCost(name, per_expert(name), 0.0))
            else:
                helpers.append(ExecutorCost(name, fitted[1], fitted[0]))
        return main, helpers

    def _count_rows(self, names: tuple[str, ...]) -> list:
        """The plan for one set of helpers, recomputed only when a cost has really moved."""
        from freetoken.moe.placement import plan_miss_counts

        if _FORCE_GPU_ONLY:
            return [(m,) + (0,) * len(names) for m in range(self._count_width)]
        main, helpers = self._executor_costs(list(names))
        # This device keeps at least its throughput share of the fetching (see min_main in
        # plan_miss_counts): what it fetches stays cached, what a helper computes does not.
        speeds = [1.0 / max(cost.per_expert_seconds, 1e-9) for cost in (main, *helpers)]
        share = speeds[0] / sum(speeds)
        min_main = [int(share * m + 0.5) for m in range(self._count_width)]
        quantum = 20e-6  # costs jitter step to step; a plan that followed the jitter would flap
        signature = tuple(
            round(min(value, 1.0) / quantum)
            for cost in (main, *helpers)
            for value in (cost.fixed_seconds, cost.per_expert_seconds)
        ) + tuple(min_main)
        cached = self._count_plan_memo.get(names)
        if cached is not None and cached[0] == signature:
            return cached[1]
        rows = plan_miss_counts(main, helpers, self._count_width - 1, min_main=min_main)
        self._count_plan_memo[names] = (signature, rows)
        return rows

    def _write_count_rows(self, layers: list[int], names: tuple[str, ...], rows: list) -> None:
        table = torch.tensor(rows, dtype=torch.int32)  # [width, 1 + helpers]
        index = torch.tensor(layers, dtype=torch.long)
        self._fetch_table_host[index] = table[:, 0]
        if names:
            bounds = torch.cumsum(table[:, 1:], dim=1, dtype=torch.int32).T.contiguous()
            self._helper_bounds_host[index, : len(names)] = bounds

    def plan_counts_for(self, layer_id: int, names: list[str]) -> None:
        """An eager step's plan for this layer, written where the device reads it."""
        key = tuple(names)
        self._count_layers[layer_id] = key
        self._write_count_rows([layer_id], key, self._count_rows(key))
        self._count_rows_last.pop(key, None)  # the next refresh rewrites every such layer

    def _refresh_count_placement(self) -> None:
        """Once per replayed step: rewrite the rows of every layer whose plan changed."""
        groups: dict[tuple[str, ...], list[int]] = {}
        for layer_id, key in self._count_layers.items():
            groups.setdefault(key, []).append(layer_id)
        for key, layers in groups.items():
            rows = self._count_rows(key)
            if self._count_rows_last.get(key) is rows:
                continue
            self._write_count_rows(layers, key, rows)
            self._count_rows_last[key] = rows

    def assign_overflow_counts(self, layer_id: int, overflow: torch.Tensor, names: list[str]) -> dict:
        """Deal this layer-step's overflow routes to the helpers by count, on the device.

        Routes are ranked in order among the overflow positions, and helper ``i`` takes the
        ranks between the row's cumulative bounds for this miss count. The last helper takes
        every rank from its lower bound on, so each overflow route is assigned exactly once
        even when the counts disagree with the routes (a batch where two tokens miss the
        same expert).
        """
        self._helper_bounds_dev[layer_id].copy_(self._helper_bounds_host[layer_id], non_blocking=True)
        flat = overflow.reshape(-1)
        rank = torch.cumsum(flat.to(torch.int32), dim=0) - 1
        row = self.num_missing_full.reshape(-1)[:1].clamp(max=self._count_width - 1)
        bounds = self._helper_bounds_dev[layer_id].index_select(1, row).reshape(-1)
        assignment, lower = {}, None
        last = len(names) - 1
        for index, name in enumerate(names):
            mask = flat if lower is None else flat & (rank >= lower)
            if index < last:
                upper = bounds[index]
                mask = mask & (rank < upper)
                lower = upper
            assignment[name] = mask.view(overflow.shape)
        return assignment

    def describe_count_plans(self) -> list[str]:
        lines = []
        for key in sorted(set(self._count_layers.values())):
            memo = self._count_plan_memo.get(key)
            if memo is None:
                continue
            plan = " ".join(f"{m}:" + "/".join(map(str, row)) for m, row in enumerate(memo[1]))
            costs = []
            for name in ("gpu", *key):
                fitted = self.cost_tracker.cost(name)
                if fitted is not None:
                    costs.append(f"{name} {fitted[0] * 1e3:.2f}+{fitted[1] * 1e3:.2f} ms/expert")
            lines.append(
                f"moe count plan (m: gpu/{'/'.join(key)}): {plan}"
                + (f"; fitted {', '.join(costs)}" if costs else "")
            )
        return lines

    def record_event(self):
        """A stream marker for timing device work, or ``None`` where there is no device."""
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def note_step_timing(self, fetch_start, fetch_end, gemm_end) -> None:
        """Queue this step's device timings, and harvest an older step's if it has landed.

        The fetch is enqueued on the stream and returns before a byte has moved, so wall
        clock around it measures the launch and reports a link hundreds of times faster
        than it is -- which would hand this device every miss on the strength of work it
        had not done yet. Stream events measure the work itself, but reading one costs a
        synchronisation, and synchronising here would destroy the overlap the split exists
        to create. So the reading is deferred: a step's events are collected once a later
        step finds them already complete, which never blocks and is a step or two behind --
        far finer than the rates themselves move.

        The fetched count has the same problem and the same answer: it is copied to pinned
        host memory on the stream and read when the events say the copy has landed.
        """
        if fetch_end is None or gemm_end is None:
            return
        if not self._fetched_staging:
            # A ring rather than one buffer: several steps can be in flight, and each needs
            # its own landing place. Pinned, because the copy has to be able to ride the
            # stream -- a pageable destination would make it synchronous and put the
            # blocking wait back exactly where this is trying to avoid it.
            self._fetched_staging = [
                torch.empty_like(self.num_indices, device="cpu").pin_memory()
                for _ in range(_TIMING_RING)
            ]
        staged = self._fetched_staging[self._timing_slot]
        self._timing_slot = (self._timing_slot + 1) % _TIMING_RING
        staged.copy_(self.num_indices, non_blocking=True)
        # Gate on an event recorded *after* the count copy, not on the GEMM's. The copy is
        # enqueued behind the GEMM, so the GEMM finishing says nothing about whether the
        # count has landed -- reading on that signal would pair this step's timings with a
        # previous step's count, silently and only sometimes.
        ready = torch.cuda.Event()
        ready.record()
        self._pending_timings.append((fetch_start, fetch_end, gemm_end, staged, ready))

        while self._pending_timings and self._pending_timings[0][4].query():
            start, end, gemm, count, _ = self._pending_timings.pop(0)
            fetched, seconds = int(count.item()), start.elapsed_time(end) / 1e3
            self.rate_tracker.observe("gpu", fetched, seconds, self.bytes_per_expert)
            # The fit separates the copy's launch from its cost per expert: rated as a plain
            # average, a step that fetched one expert reads 3x slow and the plan gives this
            # device less, which reads slower still -- until it fetches nothing at all.
            if not self._fetch_calibrated:
                self.cost_tracker.observe("gpu", 1, fetched, seconds)
            self._gpu_busy_seconds = end.elapsed_time(gemm) / 1e3
        if len(self._pending_timings) >= _TIMING_RING:
            # Nothing is completing, and the ring is about to be reused underneath entries
            # that have not been read. Drop the oldest rather than report a count that
            # belongs to a different step.
            del self._pending_timings[: len(self._pending_timings) - _TIMING_RING + 1]

    def split_shares(self, layer_id: int, helper_names: list[str]) -> dict[str, float]:
        """Fraction of this step's misses each executor should take, this device included.

        The main device is named ``"gpu"`` and is always in the split: it is the one
        executor that is always present, and the fraction it gets is what the capped-fetch
        kernel is told to fetch.

        ``layer_id`` does not enter the answer; :meth:`refresh_placement` relies on that to
        divide once per set of helpers rather than once per layer.
        """
        from freetoken.moe.placement import split_misses

        self._drain_self_timed()
        names = ["gpu", *helper_names]
        if _FORCE_GPU_ONLY:
            # Diagnostic: keep every executor in the wiring but give the work to this
            # device, so a wrong answer can be attributed to the split itself rather than
            # to what the other executors did with their share.
            return {name: (1.0 if name == "gpu" else 0.0) for name in names}
        # Measure this device's fetch directly, once, the same way the counts path does.
        # Self-guarded, and refresh_placement -- the only caller -- runs outside the graph.
        self.calibrate_fetch_cost()
        # The GEMM this device owes regardless of the split is time it starts the step
        # already committed to, so it is given proportionally fewer misses.
        busy = getattr(self, "_gpu_busy_seconds", 0.0)
        if self._fetch_calibrated:
            # ...but only when that figure means something. It is written by
            # note_step_timing, which lives inside the captured region and so never runs
            # again once decode is driven by replays: what it holds was measured in the
            # eager warm-up, the very window calibrate_fetch_cost exists to distrust --
            # there one expert read 9.4 ms against a real 0.7. An inflated "already
            # committed" figure makes this device look busier than it is and pushes misses
            # onto the helpers; on tm that showed as the iGPU being the long pole (2.68 ms
            # against the dGPU's 1.9) while the dGPU sat at 37 % utilisation. Omitting an
            # unknown is better than using a wrong one.
            busy = 0.0
        rates = self.rate_tracker.rates(names, busy={"gpu": busy})
        # A large nominal count keeps rounding out of the ratio; the kernel and the route
        # assignment both work in fractions, so only the proportions matter here.
        placement = split_misses(rates, 1024, self.bytes_per_expert)
        return {name: placement.counts[name] / 1024 for name in names if name in placement.counts}

    def _drain_self_timed(self) -> None:
        """Fold in what the executors that measure themselves have measured.

        Their samples are the only ones a captured step produces -- the engine's own
        clock cannot run inside a replay -- so without this the split a graph was captured
        with is the split it keeps for the rest of the run.
        """
        freeze = os.environ.get("FREETOKEN_DEVICE_FREEZE_RATE", "") == "1"
        timed = [(f"gpu{executor.device_index}", executor) for executor in self.device_executors]
        if self.cpu_executor is not None and getattr(self.cpu_executor, "self_timed", False):
            timed.append(("cpu", self.cpu_executor))
        for name, executor in timed:
            # always drained, so they cannot pile up; (tasks, routes, seconds) where the
            # executor keeps totals, one task per sample where it times each layer
            take_tasks = getattr(executor, "take_task_samples", None)
            if take_tasks is not None:
                samples = take_tasks()
            else:
                samples = [(1, routes, seconds) for routes, seconds in executor.take_samples()]
            if freeze:
                # Diagnostic: keep this executor's rate -- and with it the division --
                # where it stood, so a step never sees the placement move under it.
                continue
            for tasks, routes, seconds in samples:
                self.rate_tracker.observe(name, routes, seconds, self.bytes_per_expert)
                self.cost_tracker.observe(name, tasks, routes, seconds)

    def _skips_movement(self, layer_id: int) -> bool:
        """Layers whose host bank this cache can never read: the pages are gone.

        Only resident layers qualify -- their bank was uploaded and released. A worker
        layer's bank is still there and still valid, so this device fetches the share of it
        the placement leaves here, which is what makes the layer shared rather than
        surrendered."""
        return layer_id in self.resident_layer_ids

    def _skips_prefill_stream(self, layer_id: int) -> bool:
        """Layers the prefill double buffer must not stream.

        Only resident layers: their bank was uploaded and released, so there is nothing to
        stream. A layer a worker can serve is streamed like any other -- the worker takes a
        share of decode, not ownership of the layer, and prefill goes the ordinary way.
        """
        return layer_id in self.resident_layer_ids

    def resident_views(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """This layer's VRAM expert banks in registration order; row == expert id."""
        return tuple(self.resident_banks[name][layer_id] for name in self.bank_schema)

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")
        if self._skips_prefill_stream(layer_id):
            # A resident layer's host source is a released mmap -- prefetching it would copy
            # dropped (zero) pages into the buffer -- and a worker-served layer's prefill is
            # answered by the worker, so the buffer would be filled and never released.
            # Guarding here rather than at the call site also covers the look-ahead
            # prefetch of layer_id + 1 in OffloadMoELayer._wait_prefill_overlap.
            return

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts(self, layer_id, expert_ids)

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        self._hybrid_ensure_ran = True
        # The copy is issued here, inside whatever region the caller is in: under capture it
        # becomes a node reading a fixed pinned address, which is what lets a later step
        # change the fraction without recapturing.
        self._fetch_frac_dev.copy_(self._fetch_frac_host, non_blocking=True)
        table = None
        if self.placement_counts and self._fetch_table_dev is not None:
            self._fetch_table_dev[layer_id].copy_(self._fetch_table_host[layer_id], non_blocking=True)
            table = self._fetch_table_dev
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self._fetch_frac_dev,
            fetch_table=table,
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.miss_hist.zero_()

    def record_miss_hist(self, layer_id: int) -> None:
        """Count this layer-step's fetch into :attr:`miss_hist` (``FREETOKEN_MISS_HIST``)."""
        index = self.num_indices.clamp(max=_MISS_HIST_BINS - 1)
        self.miss_hist[layer_id].index_add_(0, index, self._miss_hist_one)

    def miss_hist_summary(self) -> dict | None:
        """Share of layer-steps by fetched-expert count over every layer, or None if empty."""
        totals = self.miss_hist.sum(0).tolist()
        steps = sum(totals)
        if not steps:
            return None
        last = max(i for i, count in enumerate(totals) if count)
        return {"steps": steps, "share": [count / steps for count in totals[: last + 1]]}

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def _counted_by_hybrid_kernel(self) -> bool:
        """Which set of counters holds this run's numbers.

        The capped-fetch kernel writes its own, and it now runs on any layer whose misses
        are shared with another executor -- which is decided per layer, not by the backend
        name. Reading the wrong set reports zero steps and silently hides everything
        downstream of it, which is how this first went unnoticed.
        """
        return self.decode_target == "hybrid" or self._hybrid_ensure_ran

    def decode_miss_stats(self) -> dict:
        if self._counted_by_hybrid_kernel():
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self._counted_by_hybrid_kernel():
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.lru_slots / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self.inplace_layer_ids:
            # Nothing is ever staged for one of these: its fetch fraction is held at zero
            # because a miss goes to the worker that reads the bank in place. Reaching here
            # with nothing to copy is the normal case, not an error.
            return
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            if self.quant_format == "gguf":
                for view, per_layer in zip(self.layer_bank_views(layer_id, self.num_experts),
                                           (p for p, _ in self.banks)):
                    view.copy_(per_layer[layer_id])
                return
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            mixed = self._copy_layer_feat_bytes is not None
            fast_index_copy_multi_jit(
                self._copy_layer_dst_ptrs[layer_id] if mixed else self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_layer_feat_bytes[layer_id] if mixed else self._copy_feat_bytes,
                self.evict_slots,
                self.src_indices,
                self.num_indices,
                dst_stride_bytes=self._copy_layer_dst_stride[layer_id] if mixed else None,
            )
            return

        if self.quant_format == "gguf":
            raise RuntimeError("gguf MoE banks need the fused copy (FREETOKEN_FUSED_COPY=0?)")
        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                per_layer[layer_id],
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
