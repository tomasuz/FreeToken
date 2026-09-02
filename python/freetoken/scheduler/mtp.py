"""Draft-1 MTP (nextn) self-speculative decode for the server decode path.

This is the scheduler session: the layer between the engine's ``--mtp`` head (loaded,
resident, but never run) and a real speculative generation loop. It mirrors the
validated in-process loop (repo root ``freetoken-mtp-spec-demo.py``, PASS byte-identical
vs greedy) but INCREMENTAL: KV pages and GDN recurrent/conv state are kept across steps,
so every verify is a 2-token continuation of the request's cached prefix instead of a
full re-prefill. A rejected draft rolls the GDN state back to the after-committed-token
boundary (the round-1 ``FLAMetadata.mtp_boundary_dst`` snapshot) and frees the draft's
tail KV page, keeping the ~1.4x win of the fused 2-token verify.

Activation — all must hold, otherwise the normal scheduler path is untouched:
  * engine config ``--mtp`` (model carries the MTP head) and a plain (bf16) lm_head
  * env ``FREETOKEN_MTP_SPEC=1``
  * KV ``page_size == 1``   (a reject frees exactly the draft's page)
  * exactly ONE decoding request, and it samples greedily
  * no pending prefill / rebuild and no in-flight overlapped batch (empty pipeline)

Greedy-only by design: a draft is verified against the base argmax, so emitted tokens
are byte-identical to plain greedy generation regardless of draft quality; a wrong
draft is simply rejected. The request's committed state (cached_len / device_len /
input_ids) always satisfies the normal decode invariant at a step boundary, so MTP mode
can yield to (or resume from) the normal decode loop at any token without special
handling.

The head draft is a single-row MTP-head forward (its own appended full-attention layer).
For one causal row the head only needs its own freshly-written KV, so the row reuses the
request's position-0 page (layer-40 KV is head-private and never read by the base
model); past head context is not maintained, matching the validated demo's L=1 drafts
(affects acceptance rate only, never output correctness).
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from freetoken.core import Batch, Req
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Env gate: FREETOKEN_MTP_SPEC=1 turns the --mtp head into an actual spec loop.
ENV_SPEC = "FREETOKEN_MTP_SPEC"


class MTPDecodeMixin:
    """Mixed into ``Scheduler``. All methods run on the scheduler's current stream
    (self.stream under overlap mode, engine.stream under DISABLE_OVERLAP), which is
    safe because the MTP drive only takes over when the overlap pipeline is empty and
    runs synchronously (launch + decide + rollback inside one iteration)."""

    # once a step fails we disable spec for the process and let normal decode resume
    _mtp_disabled = False
    # uid -> {"draft": int | None} (pending head draft for the next verify step)
    _mtp_state: dict = {}

    # ------------------------------------------------------------------ gating
    def _mtp_configured(self) -> bool:
        if getattr(self, "_mtp_disabled", False):
            return False
        try:
            cfg = self.engine.config
            model = self.engine.model
        except Exception:  # noqa: BLE001
            return False
        if not (getattr(cfg, "mtp", False) and getattr(model, "mtp", None) is not None):
            return False
        if os.environ.get(ENV_SPEC) != "1":
            return False
        if self.cache_manager.page_size != 1:
            return False
        # verify needs argmax logits at BOTH new positions -> plain bf16 lm_head weight
        if self._mtp_lmhead_weight() is None:
            return False
        return True

    def _mtp_lmhead_weight(self):
        """Shared vocab matrix of the lm_head (tied-aware). None for quantized heads
        (NVFP4), which the spec path cannot score row-wise and so stays disabled."""
        lm = getattr(self.engine.model, "lm_head", None)
        if lm is None:
            return None
        return getattr(getattr(lm, "tied_embedding", None) or lm, "weight", None)

    def _mtp_target(self) -> Req | None:
        """The single greedy decoding request to speculate, else None."""
        if not self._mtp_configured():
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
        if req.linear_slot_idx is None:
            return None
        return req

    def _mtp_takeover(self, data=None) -> bool:
        """True when the spec drive can take over the scheduler loop right now (the
        overlap pipeline must be empty: ``data`` is the not-yet-drained batch)."""
        return data is None and self._mtp_target() is not None

    # ------------------------------------------------------------------ drive
    def _mtp_drive(self) -> None:
        """Run synchronous spec decode for the single request until it finishes, is
        aborted, or the scheduler gains other work (a new prompt / second request),
        at which point the normal loop resumes seamlessly from the req's valid state."""
        req = self._mtp_target()
        if req is None:
            return
        self._mtp_state[req.uid] = {"draft": None}
        # stats for this drive: forwards = base forwards run, committed = real tokens
        # emitted, verified/accepted = drafts checked / accepted. tokens/base-forward is
        # the headline speedup signal (plain greedy = 1.00).
        stat = self._mtp_stat = {"forwards": 0, "committed": 0, "verified": 0, "accepted": 0}
        t0 = time.monotonic()
        logger.info_rank0(f"[mtp] speculative decode engaged uid={req.uid} cached_len={req.cached_len}")
        try:
            while True:
                for msg in self.receive_msg(blocking=False):
                    self._process_one_msg(msg)
                if not self._mtp_takeover():
                    break
                req = self._mtp_target()
                st = self._mtp_state.setdefault(req.uid, {"draft": None})
                done = self._mtp_step(req, st)
                self._flush_abort_acks()
                if stat["forwards"] and stat["forwards"] % 40 == 0:
                    dt = max(time.monotonic() - t0, 1e-9)
                    logger.info_rank0(
                        "[mtp] progress: committed %d tok in %d base-forward(s) = %.2f "
                        "tok/forward, alpha %d/%d = %.2f, %.1f tok/s",
                        stat["committed"], stat["forwards"],
                        stat["committed"] / stat["forwards"],
                        stat["accepted"], stat["verified"],
                        (stat["accepted"] / stat["verified"]) if stat["verified"] else 0.0,
                        stat["committed"] / dt,
                    )
                if done:
                    break
        except Exception:  # noqa: BLE001 -- dev bring-up: disable, resume normal decode
            # print the traceback explicitly: this server's logging formatter drops the
            # exc_text that logger.exception attaches, so it would otherwise be invisible
            import traceback

            traceback.print_exc()
            logger.exception(
                "[mtp] speculative decode step failed; disabling MTP spec "
                "(normal decode resumes from the request's committed state)"
            )
            self._mtp_disabled = True
        if stat["forwards"]:
            dt = max(time.monotonic() - t0, 1e-9)
            logger.info_rank0(
                "[mtp] disengaged: committed %d tok in %d base-forward(s) = %.2f tok/forward "
                "(plain greedy = 1.00), alpha %d/%d = %.2f, %.1f tok/s",
                stat["committed"], stat["forwards"], stat["committed"] / stat["forwards"],
                stat["accepted"], stat["verified"],
                (stat["accepted"] / stat["verified"]) if stat["verified"] else 0.0,
                stat["committed"] / dt,
            )
        else:
            logger.info_rank0("[mtp] speculative decode disengaged (no spec steps ran)")

    def _mtp_step(self, req: Req, st: dict) -> bool:
        """One spec iteration: seed (no draft) or verify (draft). Returns True when the
        request finished (removed) during this step."""
        stat = self._mtp_stat
        draft = st.get("draft")
        if draft is None:
            committed = self._mtp_seed(req, st)
        else:
            stat["verified"] += 1
            committed, accepted = self._mtp_verify(req, st, draft)
            if accepted:
                stat["accepted"] += 1
        stat["forwards"] += 1
        stat["committed"] += committed
        return req not in self.decode_manager.running_reqs or req in self.finished_reqs

    # ------------------------------------------------------- continuation forward
    def _mtp_run_extend(self, req: Req, tokens: list, boundary_slot: int | None = None):
        """Eagerly process ``tokens`` (1..2 ids) as a continuation of ``req`` at
        positions [C, C+len) where C = req.cached_len (the newest committed real token,
        unprocessed). Temporarily bumps ``req.device_len`` to C+len so the scheduler's
        batch prep allocates the new KV pages / builds continuation metadata.

        Returns the base model's post-final-norm hidden ``[len(tokens), hidden]``. On
        failure restores ``req.device_len`` and frees the just-allocated tail pages so
        normal decode can resume cleanly, then re-raises (drive disables spec).

        ``boundary_slot`` (verify only, default None): GDN layers snapshot the
        after-first-token (after-committed-``u``) recurrent+conv state there for the
        reject rollback.
        """
        C = req.cached_len
        n = len(tokens)
        saved = req.device_len
        try:
            req.device_len = C + n
            b = Batch(reqs=[req], phase="prefill")
            for j, t in enumerate(tokens):
                self.token_pool[req.table_idx, C + j] = int(t)
            fi = self._prepare_batch(b)
            if boundary_slot is not None:
                b.fla_metadata.mtp_boundary_dst = torch.tensor(
                    [boundary_slot], dtype=torch.int64, device=self.device
                )
            b.input_ids = self.token_pool[fi.input_tuple[0]]
            with self.engine.ctx.forward_batch(b):
                hidden = self.engine.model.forward_hidden()  # [n, hidden]
            return hidden
        except Exception:  # noqa: BLE001
            # Restore a clean normal-decode state: drop the speculative tail pages and
            # the temp device_len bump so the req's committed token at cached_len is
            # simply re-processed by the normal decode path.
            try:
                req.device_len = saved
                self.cache_manager.free_tail_pages(req, keep_len=C)
            except Exception:  # noqa: BLE001
                pass
            raise

    # ---------------------------------------------------------------- commits
    def _mtp_stage(self, req: Req, tok: int) -> None:
        """Append ``tok`` as a committed real token (CPU ids + GPU token pool) and keep
        ``device_len == len(input_ids)`` (cached_len is finalized by the step)."""
        req.append_host(torch.tensor([tok], dtype=req.input_ids.dtype))
        req.device_len = req.input_ids.numel()
        self.token_pool[req.table_idx, req.device_len - 1] = int(tok)

    def _mtp_emit(self, req: Req, tokens: list, kept_processed: int) -> None:
        """Commit real tokens in order, mirroring the normal drain's finish rules
        (length / eos / stop-strings) and emitting one DetokenizeMsg per token.
        ``kept_processed`` = this step's kept processed count = the new cached_len
        (index of the newest committed real token) once the request survives.
        A finished request is removed from the decode manager and its resources freed.
        Returns the number of tokens actually committed (emitted) this step.
        """
        msgs: list = []
        finished = False
        for tok in tokens:
            if req.input_ids.numel() >= req.max_device_len:
                finished = True  # output budget exhausted -> "length"
                break
            self._mtp_stage(req, tok)
            hit_length = not req.can_decode
            hit_eos = not req.sampling_params.ignore_eos and tok in self.eos_token_ids
            matched_stop = (
                self._match_stop_str(req)
                if not hit_eos and req.sampling_params.stop_strs
                else None
            )
            finished = hit_length or hit_eos or matched_stop is not None
            finish_reason = (
                ("stop" if (hit_eos or matched_stop is not None) else "length")
                if finished
                else None
            )
            if (
                tok == self.toolcall_anchor_id
                and req.toolcall_anchor_len is None
                and not finished
            ):
                req.toolcall_anchor_len = req.input_ids.numel()
            msgs.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=tok,
                    finished=finished,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
            )
            if finished:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished_reqs.add(req)
                break

        if not finished:
            req.cached_len = kept_processed
            assert req.device_len == req.input_ids.numel() == kept_processed + 1, (
                f"[mtp] bad commit state cached={req.cached_len} device={req.device_len} "
                f"ids={req.input_ids.numel()}"
            )
        if msgs:
            used, total = self._kv_usage_pages()
            mamba = self._mamba_slot_usage()
            swa = self._swa_token_usage()
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba or (0, 0)
            swa_used, swa_total = swa or (0, 0)
            for m in msgs:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
            self.send_result(msgs)
        return len(msgs)

    # -------------------------------------------------------------- head draft
    def _mtp_head_draft(self, req: Req, prev_hidden: torch.Tensor) -> int | None:
        """Draft the token after the newest committed real token
        (``req.input_ids[req.cached_len]``) from ``prev_hidden`` = the base hidden of the
        token just before it. Single-row engine MTP-head forward over the request's
        position-0 page (head layer-40 KV is head-private). Returns the argmax token id,
        or None if the head call fails (drive disables spec; output stays correct)."""
        model = self.engine.model
        head = getattr(model, "mtp", None)
        if head is None:
            return None
        nid = int(req.input_ids[req.cached_len].item())
        ids = torch.tensor([nid], dtype=req.input_ids.dtype, device=self.device)
        pos = req.cached_len
        try:
            hr = Req(
                input_ids=torch.zeros(1, dtype=req.input_ids.dtype),
                table_idx=req.table_idx,
                cached_len=0,
                output_len=1,
                uid=req.uid,
                sampling_params=req.sampling_params,
                cache_handle=req.cache_handle,
            )
            b = Batch(reqs=[hr], phase="prefill")
            b.padded_reqs = [hr]
            b.input_ids = ids
            b.positions = torch.tensor([pos], dtype=torch.int32, device=self.device)
            b.out_loc = self.engine.page_table[req.table_idx, 0:1]
            self.engine.attn_backend.prepare_metadata(b)
            with self.engine.ctx.forward_batch(b):
                logits = model.forward_mtp(prev_hidden, ids)  # [1, vocab]
            return int(torch.argmax(logits, dim=-1).item())
        except Exception:  # noqa: BLE001 -- dev bring-up; fall back to normal decode
            logger.exception("[mtp] MTP head draft failed; disabling MTP spec")
            self._mtp_disabled = True
            return None

    # ---------------------------------------------------------------- steps
    def _mtp_seed(self, req: Req, st: dict) -> int:
        """Seed: no pending draft. Process the newest committed real token ``u`` alone
        (1-token continuation), commit its real successor, then draft the token after it.
        (This is the demo's Case A, done once per MTP engagement.) Returns the number of
        tokens committed."""
        C = req.cached_len
        u = int(req.input_ids[C].item())
        hidden = self._mtp_run_extend(req, [u])  # [1, H]
        if hidden is None:
            return 0
        weight = self._mtp_lmhead_weight()
        logits = F.linear(hidden, weight)  # [1, vocab]
        real = int(torch.argmax(logits, dim=-1).item())
        committed = self._mtp_emit(req, [real], kept_processed=C + 1)
        if req in self.finished_reqs:
            st["draft"] = None
            return committed
        # draft the token after `real`: prev_hidden = hidden of u (its predecessor)
        st["draft"] = self._mtp_head_draft(req, hidden[0:1])
        return committed

    def _mtp_verify(self, req: Req, st: dict, draft: int) -> tuple:
        """Verify a pending draft: fused 2-token continuation [u@C, d@C+1]. logits@C
        verify d:
          accept -> commit d and the base's own next token c (=argmax@C+1); roll nothing
                    back (live GDN state is already after d, which is kept); the next
                    draft comes from hidden[d] + c.
          reject -> commit only the real token; roll the GDN state back to after-u (the
                    round-1 boundary snapshot in the spare slot) and free d's KV page;
                    the next draft comes from hidden[u] + real.
        Returns (committed_tokens, accepted_bool).
        """
        C = req.cached_len
        u = int(req.input_ids[C].item())
        pool = self.engine.linear_state_pool
        spare = pool.alloc(1)[0]
        try:
            hidden = self._mtp_run_extend(req, [u, draft], boundary_slot=spare)  # [2, H]
            weight = self._mtp_lmhead_weight()
            logits = F.linear(hidden, weight)  # [2, vocab]
            real = int(torch.argmax(logits[0:1], dim=-1).item())
            if real == draft:
                c = int(torch.argmax(logits[1:2], dim=-1).item())
                committed = self._mtp_emit(req, [draft, c], kept_processed=C + 2)
                if req in self.finished_reqs:
                    st["draft"] = None
                    return committed, True
                # newest committed = c@C+2; its predecessor d was the last kept process
                st["draft"] = self._mtp_head_draft(req, hidden[1:2])
                return committed, True
            else:
                committed = self._mtp_emit(req, [real], kept_processed=C + 1)
                if req in self.finished_reqs:
                    st["draft"] = None
                    return committed, False
                # reject: drop the draft's tail KV page(s) and roll GDN back to after-u
                self.cache_manager.free_tail_pages(req, keep_len=C + 1)
                pool.copy_from(spare, req.linear_slot_idx)
                # newest committed = real@C+1; its predecessor u was the kept process
                st["draft"] = self._mtp_head_draft(req, hidden[0:1])
                return committed, False
        finally:
            pool.free(spare)


__all__ = ["MTPDecodeMixin"]
