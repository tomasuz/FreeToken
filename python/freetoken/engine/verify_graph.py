"""Captured MTP verify forwards: one request, T consecutive tokens, every row scored.

A verify is an extend of 1..4 tokens (``scheduler/mtp4.py``; 1: a step with no draft). Run eagerly it costs about three
graph decode steps on the GPU it was measured on, all of it launch overhead, so each T gets its own graph.
The forward is the ordinary prefill-phase one with ``batch.mtp_verify`` (the GDN layers take
the decode kernels row by row) and ``batch.mtp_capture`` (all-row logits + the wide residual),
bound to static buffers:

* the token ids, positions and KV destinations of the T rows;
* the request's linear-state slot (GDN / conv / PLE state);
* the QSA addressing of one request: kv length, pending-ring slot and block table.

The MoE layers take their decode path for T <= 4 tokens, which is the capturable one; the disk
PLE table is staged into its graph buffers before the replay, as a graph decode step does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

from freetoken.core import Batch, Req, get_global_ctx
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .engine import Engine

logger = init_logger(__name__)


class VerifyGraphs:
    def __init__(self, engine: "Engine", sizes=(2, 3, 4)) -> None:
        from freetoken.attention.qsa_sparse import QSASparseAttnBackend

        self.engine = engine
        self.sizes = tuple(sorted(sizes))
        self.device = engine.device
        backend = engine.attn_backend
        assert isinstance(backend, QSASparseAttnBackend), "verify graphs serve the QSA backend"
        self.backend = backend
        max_t = max(self.sizes)
        dev = self.device
        i32 = dict(dtype=torch.int32, device=dev)
        page_table = get_global_ctx().page_table
        self.input_ids = torch.zeros(max_t, **i32)
        self.positions = torch.zeros(max_t, **i32)
        self.out_loc = torch.zeros(max_t, **i32)
        self.slot = torch.zeros(1, **i32)  # linear-state slot
        self.ring_slot = torch.zeros(1, **i32)  # table_idx: QSA pending ring
        self.kv_len = torch.zeros(1, **i32)
        self.block_table = torch.zeros(1, -(-page_table.shape[1] // backend.page_size), **i32)
        self.token_to_req = torch.zeros(max_t, **i32)
        self.has_init = torch.ones(1, dtype=torch.bool, device=dev)
        self.cu = {t: torch.tensor([0, t], **i32) for t in self.sizes}
        self.last = {t: torch.tensor([t - 1], **i32) for t in self.sizes}
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: Dict[int, torch.Tensor] = {}
        self.residual: Dict[int, torch.Tensor] = {}
        self._batches: Dict[int, Batch] = {}

    # ------------------------------------------------------------------ batch
    def _batch(self, T: int, req: Req) -> Batch:
        from freetoken.attention.linear import FLAMetadata
        from freetoken.attention.qsa_sparse import QSASparseMetadata

        b = Batch(reqs=[req], phase="prefill")
        b.padded_reqs = [req]
        b.input_ids = self.input_ids[:T]
        b.positions = self.positions[:T]
        b.out_loc = self.out_loc[:T]
        b.mtp_capture = True
        b.mtp_verify = True
        b.fla_metadata = FLAMetadata(
            cu_seqlens=self.cu[T], cache_indices=self.slot, has_initial_state=self.has_init
        )
        pin = dict(dtype=torch.int32, device="cpu", pin_memory=True)
        b.attn_metadata = QSASparseMetadata(
            is_decode=False,
            last_indices=self.last[T],
            qo_indptr_cpu=torch.tensor([0, T], **pin),
            kv_len_cpu=torch.tensor([T], **pin),
            token_to_req=self.token_to_req[:T],
            cu_seqlens=self.cu[T],
            seq_lens=self.kv_len,
            ring_slots=self.ring_slot,
            block_table=self.block_table,
        )
        return b

    def _stage(self, table_idx: int, slot: int, kv_len: int) -> None:
        backend = self.backend
        rows = torch.tensor([table_idx], dtype=torch.int64, device=self.device)
        self.block_table.copy_(backend._block_table(rows))
        self.ring_slot.fill_(table_idx)
        self.slot.fill_(slot)
        self.kv_len.fill_(kv_len)

    # ---------------------------------------------------------------- capture
    def capture(self) -> None:
        engine = self.engine
        model = engine.model
        dummy = engine.dummy_req
        cache = engine.moe_offload_cache
        stream = engine.stream
        executors = engine.graph_runner._device_executors()
        assert not executors, "verify graphs do not drive helper devices yet"
        pool = torch.cuda.graph_pool_handle()
        dummy_slot = dummy.linear_slot_idx if dummy.linear_slot_idx is not None else dummy.table_idx
        with torch.cuda.stream(stream):
            for T in sorted(self.sizes, reverse=True):
                # a stand-in request: T fresh tokens on the dummy's row and state slot
                req = Req(
                    input_ids=torch.zeros(T, dtype=torch.int32),
                    table_idx=dummy.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=dummy.sampling_params,
                    cache_handle=dummy.cache_handle,
                )
                req.linear_slot_idx = dummy.linear_slot_idx
                req.device_len = T
                self.input_ids.zero_()
                self.positions[:T].copy_(torch.arange(T, dtype=torch.int32, device=self.device))
                self.out_loc.zero_()
                self._stage(dummy.table_idx, dummy_slot, T)
                b = self._batch(T, req)
                g = torch.cuda.CUDAGraph()
                with get_global_ctx().forward_batch(b), model.forward_host_ctx(b, False):
                    model.forward()  # warm: autotune, scratch buffers, memoized indices
                with get_global_ctx().forward_batch(b):
                    with torch.cuda.graph(g, pool=pool, stream=stream):
                        logits = model.forward()
                self.graphs[T] = g
                self.logits[T] = logits
                self.residual[T] = model.model._mtp_residual
                self._batches[T] = b
                if cache is not None:
                    cache.reset()
        torch.cuda.synchronize(self.device)
        logger.info_rank0(f"MTP verify graphs captured for T in {list(self.sizes)}")

    # ----------------------------------------------------------------- replay
    def run(self, fwd: Batch, req: Req, slot: int, tokens: list[int]):
        """Replay the T = len(tokens) graph for ``req`` at positions [C, C+T), where ``fwd`` is
        the scheduler's prepared batch (its out_loc / positions / input ids). Returns the
        captured (logits [T, vocab], residual [T, hc*H])."""
        from freetoken.models.qwen4_exp.ple_disk import _context

        T = len(tokens)
        C = req.cached_len
        self.input_ids[:T].copy_(fwd.input_ids.to(torch.int32), non_blocking=True)
        self.positions[:T].copy_(fwd.positions.to(torch.int32), non_blocking=True)
        self.out_loc[:T].copy_(fwd.out_loc.to(torch.int32), non_blocking=True)
        self._stage(req.table_idx, slot, C + T)
        model = self.engine.model
        table = getattr(model, "_ple_table", None)
        if table is not None and hasattr(table, "fill"):
            run = torch.cat((
                torch.tensor(_context(req.input_ids, C, table.eos_token_id), dtype=torch.int64),
                torch.tensor(tokens, dtype=torch.int64),
            ))
            table.fill([run], graph=True)
        cache = self.engine.moe_offload_cache
        if cache is not None:
            cache.refresh_placement()
        self.graphs[T].replay()
        return self.logits[T], self.residual[T]


class HeadGraphs:
    """The MTP head's draft forward, captured for the 1..3 rows a step feeds it (see
    ``scheduler/mtp4.py``: one pair per accepted draft plus one). Eager it is ~7 ms of
    launches."""

    def __init__(self, engine: "Engine", rows=(1, 2, 3)) -> None:
        model = engine.model
        cfg = model._config
        self.engine = engine
        self.rows = tuple(sorted(rows))
        n = max(self.rows)
        width = cfg.hidden_size * cfg.qwen4_args.hc_count
        dev = engine.device
        self.R = torch.zeros(n, width, dtype=torch.bfloat16, device=dev)
        self.ids = torch.zeros(n, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(n, dtype=torch.int64, device=dev)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: Dict[int, torch.Tensor] = {}
        self.residual: Dict[int, torch.Tensor] = {}

    def capture(self) -> None:
        engine = self.engine
        model = engine.model
        stream = engine.stream
        pool = torch.cuda.graph_pool_handle()
        b = Batch(reqs=[engine.dummy_req], phase="decode")
        with torch.cuda.stream(stream), get_global_ctx().forward_batch(b):
            for n in sorted(self.rows, reverse=True):
                self.pos[:n].copy_(torch.arange(n, device=self.pos.device))
                model.mtp_forward(self.R[:n], self.ids[:n], self.pos[:n])  # warm
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    logits, R = model.mtp_forward(self.R[:n], self.ids[:n], self.pos[:n])
                self.graphs[n] = g
                self.logits[n] = logits
                self.residual[n] = R
        torch.cuda.synchronize(engine.device)
        # the warm runs wrote dummy rows into the attention ring and the head's slot cache
        model._mtp.head.self_attn.reset_ring()
        model._mtp.cache.reset()
        logger.info_rank0(f"MTP head graphs captured for {list(self.rows)} row(s)")

    def run(self, R: torch.Tensor, next_tokens: list[int], first_pos: int):
        """(argmax draft of the last row, the head residual of that row [1, hc*H]), as
        ``_mtp4_draft`` computes them eagerly. The residual lives in the graph pool: use it
        before the next replay."""
        n = len(next_tokens)
        self.R[:n].copy_(R)
        self.ids[:n].copy_(torch.tensor(next_tokens, dtype=torch.int64), non_blocking=True)
        self.pos[:n].copy_(torch.arange(first_pos, first_pos + n, dtype=torch.int64), non_blocking=True)
        cache = self.engine.model._mtp.cache
        if hasattr(cache, "refresh_placement"):
            cache.refresh_placement()
        self.graphs[n].replay()
        return int(self.logits[n][-1].argmax()), self.residual[n][-1:]


__all__ = ["HeadGraphs", "VerifyGraphs"]
