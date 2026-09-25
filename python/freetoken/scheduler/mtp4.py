"""MTP self-speculative decode for Qwen3.8 GGUF (qwen4exp) with llama.cpp's separate MTP head.

The qwen3_5 loop (``mtp.py``) needs the GDN kernel to hand out the state after the first
of two verify tokens. Qwen3.8 carries more per-slot state than the recurrence (the PLE
n-gram window, the conv windows), so this loop keeps the whole linear-state slot instead:

* before a verify the live slot is copied into a private buffer;
* the verify is a plain T-token continuation through the ordinary extend path
  (``batch.mtp_capture`` makes the model score every row and keep the wide residual);
* on a reject the buffer is copied back, the tail KV pages are freed and the verified
  tokens stay *pending*: the next verify re-processes them together with the real token
  and the next draft (T grows by one per reject, capped by ``FREETOKEN_MTP_MAX_T``).

Greedy only; emitted tokens equal plain greedy decoding whatever the drafts are. When the
loop yields to the normal scheduler it first processes all pending tokens but the newest,
so the request is back at the ordinary decode invariant (one unprocessed token).
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.core import Batch, Req
from freetoken.utils import init_logger

logger = init_logger(__name__)

# the widest verify (pending tokens + draft); past it a step runs without a draft
MAX_T = int(os.getenv("FREETOKEN_MTP_MAX_T", "4"))


class Qwen4MTPMixin:
    """Mixed into ``Scheduler`` ahead of ``MTPDecodeMixin``; defers to it for other models."""

    _mtp4_disabled = False

    # ------------------------------------------------------------------ gating
    def _mtp4_model(self):
        model = getattr(self.engine, "model", None)
        if self._mtp4_disabled or not getattr(model, "has_mtp_head", False):
            return None
        if os.environ.get("FREETOKEN_MTP_SPEC", "1") in ("0", "false", "off", "no"):
            return None
        # any page size: a verify allocates pages from cached_len like any extend, and a
        # reject frees the pages past it (free_tail_pages) before the next verify
        return model

    def _mtp4_target(self) -> Req | None:
        if self._mtp4_model() is None:
            return None
        if self.prefill_manager.runnable or self._pending_rebuild is not None:
            return None
        running = self.decode_manager.running_reqs
        if len(running) != 1:
            return None
        req = next(iter(running))
        if not getattr(req.sampling_params, "is_greedy", False):
            return None
        if req.aborted or req in self.finished_reqs or not req.can_decode:
            return None
        return req

    @staticmethod
    def _mtp4_live_slot(req: Req) -> int:
        # hybrid radix gives the request its own slot; naive keys the state by table_idx
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def _mtp4_snapshot(self, slot: int, restore: bool) -> None:
        """Save (or restore) one slot's whole linear state in a private buffer: the naive
        pool has no free slot to spare."""
        pool = self.engine.linear_state_pool
        src = [pool.conv_states, pool.recurrent_states, *pool.slot_states.values()]
        snap = getattr(self, "_mtp4_snap", None)
        if snap is None or len(snap) != len(src) or any(
            a.shape != b[:, slot].shape or a.dtype != b.dtype for a, b in zip(snap, src)
        ):
            snap = self._mtp4_snap = [torch.empty_like(t[:, slot]) for t in src]
        for buf, t in zip(snap, src):
            if restore:
                t[:, slot].copy_(buf)
            else:
                buf.copy_(t[:, slot])

    def _mtp_takeover(self, data=None) -> bool:
        if getattr(self.engine.model, "has_mtp_head", False):
            if self._mtp4_target() is None:
                return False
            if data is not None:
                # drain the in-flight decode step first: MTP runs with an empty pipeline
                self.stream.wait_stream(self.engine.stream)
                self._process_last_data(data)
                self._flush_abort_acks()
                return True  # the caller's batch is consumed; the drive returns None
            return True
        return super()._mtp_takeover(data)

    # ------------------------------------------------------------------ drive
    def _mtp_drive(self):
        if not getattr(self.engine.model, "has_mtp_head", False):
            return super()._mtp_drive()
        req = self._mtp4_target()
        if req is None:
            return None
        stat = {"forwards": 0, "committed": 0, "verified": 0, "accepted": 0, "tokens": 0}
        t0 = time.monotonic()
        logger.info_rank0(f"[mtp4] speculative decode engaged uid={req.uid} cached_len={req.cached_len}")
        draft = None
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            try:
                while True:
                    done, draft = self._mtp4_step(req, draft, stat)
                    self._flush_abort_acks()
                    if done:
                        break
                    for msg in self.receive_msg(blocking=False):
                        self._process_one_msg(msg)
                    if self._mtp4_target() is not req:
                        if req in self.decode_manager.running_reqs and req not in self.finished_reqs:
                            self._mtp4_flush(req)
                        break
            except Exception:  # noqa: BLE001 -- disable, let normal decode resume
                import traceback

                traceback.print_exc()
                logger.exception("[mtp4] speculative step failed; disabling MTP")
                self._mtp4_disabled = True
                if req in self.decode_manager.running_reqs and req not in self.finished_reqs:
                    self._mtp4_flush(req)
            self.stream.wait_stream(self.engine.stream)
        if stat["forwards"]:
            dt = max(time.monotonic() - t0, 1e-9)
            logger.info_rank0(
                "[mtp4] disengaged: %d tok in %d forwards (%.2f tok/forward, %.2f rows/forward), "
                "accepted %d/%d = %.2f, %.1f tok/s",
                stat["committed"], stat["forwards"], stat["committed"] / stat["forwards"],
                stat["tokens"] / stat["forwards"], stat["accepted"], stat["verified"],
                stat["accepted"] / max(stat["verified"], 1), stat["committed"] / dt,
            )
        self._mtp4_stat = stat
        return None

    # ---------------------------------------------------------------- forward
    def _mtp4_run(self, req: Req, tokens: list[int]):
        """Process ``tokens`` at positions [C, C+len) (C = cached_len) as one continuation.
        Returns (argmax per row [T] on the host, wide residual [T, hc*H]). Leaves
        ``req.device_len`` at C+T; the caller settles it."""
        C = req.cached_len
        T = len(tokens)
        req.device_len = C + T
        self.token_pool[req.table_idx, C : C + T] = torch.tensor(tokens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        b = Batch(reqs=[req], phase="prefill")
        fi = self._prepare_batch(b)
        b.input_ids = self.token_pool[fi.input_tuple]
        b.mtp_capture = True
        model = self.engine.model
        # the host side (PLE n-gram rows) reads the ids of every processed position, the
        # draft's included: show it the draft for the length of the forward
        n_old = req.input_ids.numel()
        buf = req._ids_buf
        buf[n_old : C + T] = torch.tensor(tokens[n_old - C :], dtype=buf.dtype)
        req.input_ids = buf[: C + T]
        try:
            with self.engine.ctx.forward_batch(b), model.forward_host_ctx(b, False):
                logits = model.forward()  # [T, vocab]
        finally:
            req.input_ids = buf[:n_old]
        top = logits.argmax(dim=-1).to("cpu").tolist()
        return top, model.model._mtp_residual

    def _mtp4_draft(self, req: Req, R: torch.Tensor, next_tokens: list[int], first_pos: int) -> int:
        """Feed (R[j], next_tokens[j]) at positions first_pos+j to the head; the argmax of the
        last row drafts the token after the newest one."""
        model = self.engine.model
        ids = torch.tensor(next_tokens, dtype=torch.int64).to(self.device, non_blocking=True)
        pos = torch.arange(first_pos, first_pos + len(next_tokens), dtype=torch.int64, device=self.device)
        b = Batch(reqs=[self.engine.dummy_req], phase="decode")
        with self.engine.ctx.forward_batch(b):
            logits, _ = model.mtp_forward(R.to(torch.bfloat16), ids, pos)
        return int(logits[-1].argmax())

    # ---------------------------------------------------------------- steps
    def _mtp4_step(self, req: Req, draft: int | None, stat: dict):
        """One verify (or a plain step when there is no draft). Returns (finished, next draft)."""
        C = req.cached_len
        pending = req.input_ids[C:].tolist()
        n_p = len(pending)
        if draft is not None and n_p + 1 > MAX_T:
            draft = None
        tokens = pending + ([draft] if draft is not None else [])
        T = len(tokens)
        live = self._mtp4_live_slot(req)
        if draft is not None:
            self._mtp4_snapshot(live, restore=False)
        try:
            t0 = time.perf_counter()
            top, R = self._mtp4_run(req, tokens)
            stat["t_verify"] = stat.get("t_verify", 0.0) + time.perf_counter() - t0
            stat["forwards"] += 1
            stat["tokens"] += T
            real = top[n_p - 1]
            if draft is not None:
                stat["verified"] += 1
            if draft is not None and real == draft:
                stat["accepted"] += 1
                seq = pending + [draft, top[n_p]]
                new, rows, kept = [draft, top[n_p]], T, C + T
            elif draft is not None:
                self._mtp4_snapshot(live, restore=True)
                self.cache_manager.free_tail_pages(req, keep_len=C)
                seq = pending + [real]
                new, rows, kept = [real], n_p, C
            else:
                seq = pending + [real]
                new, rows, kept = [real], n_p, C + n_p
        except Exception:
            if draft is not None:
                self._mtp4_snapshot(live, restore=True)
            if req.device_len > req.input_ids.numel():
                self.cache_manager.free_tail_pages(req, keep_len=C)
            req.device_len = req.input_ids.numel()
            raise
        req.device_len = req.input_ids.numel()
        committed = self._mtp_emit(req, new, kept_processed=kept, n_pending=req.input_ids.numel() + len(new) - kept)
        stat["committed"] += committed
        if req in self.finished_reqs or req not in self.decode_manager.running_reqs:
            return True, None
        # rows 0..n_p-2 fed the head last step (identical pairs); only newer rows are new
        lo = max(n_p - 1, 0)
        nxt = seq[1 : rows + 1]
        t0 = time.perf_counter()
        draft = self._mtp4_draft(req, R[lo:rows], nxt[lo:], C + lo + 1)
        stat["t_head"] = stat.get("t_head", 0.0) + time.perf_counter() - t0
        return False, draft

    def _mtp4_flush(self, req: Req) -> None:
        """Process all pending tokens but the newest (no commit): the ordinary decode
        invariant (one unprocessed token) holds again."""
        C = req.cached_len
        n = req.input_ids.numel() - C - 1
        if n <= 0:
            return
        tokens = req.input_ids[C : C + n].tolist()
        self._mtp4_run(req, tokens)
        req.cached_len = C + n
        req.device_len = req.input_ids.numel()


__all__ = ["Qwen4MTPMixin"]
