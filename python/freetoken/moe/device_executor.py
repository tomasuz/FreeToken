"""Experts computed on another GPU of this process, reading the engine's own host banks.

The sibling of :class:`~freetoken.moe.worker_executor.WorkerMoeExecutor`, and it exists
because that one should not have to be a process.

A worker is a process for exactly one reason: a device whose runtime settings differ from
this one's can only be driven from somewhere those settings are the process's own -- an
integrated GPU with no code in the build presents as an older architecture, and everything
it touches has to be found or built for that instead. A build that has code for the device
needs none of it, and then the device is just another one this process owns.

What the process boundary cost is most of the worker's machinery: the banks published as
files, mapped a second time by the child, a registration budget spent twice over. What it
did NOT cost is the flags protocol, and that is kept, because on this stack nothing else
works:

* A cross-device event aborts in the runtime (hipEventRecord -> Device::NullStream ->
  Stream::terminate).
* A capture CAN be made to span both devices -- fork on an event, join on another, and
  capture_end succeeds -- but the replay of that graph faults with an illegal access.
  Measured here, not assumed.
* While a capture is underway anywhere in the process, this device may not allocate: the
  allocator asks the driver even for a block it already has cached, and the ask is refused.

So the second device works the way the worker's child did -- it serves requests it is
told about, on its own stream, outside the graph -- and everything it runs is written into
buffers it allocated once, before any capture existed. The engine's side is four nodes a
graph records happily: three copies and a doorbell, then a wait.

Nothing is copied that should not be. The banks are read where they lie, through the
portable registration they already have; only the activation crosses, and it crosses into
host memory both devices address, never device to device.
"""

from __future__ import annotations

import os
import queue
import threading

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SLOTS = 64  # (layer, batch) pairs a decode ever asks for; one flag word each


def _nvfp4_call(banks: dict, x, w, ids, activation, act_fn, bufs=None):
    from freetoken.moe.fused_nvfp4_vec import fused_experts_nvfp4_vec

    roles = (
        ("gate_up", "gate_up_packed"), ("gate_up_scale", "gate_up_scale"),
        ("gate_up_global", "gate_up_global"), ("down", "down_packed"),
        ("down_scale", "down_scale"), ("down_global", "down_global"),
    )
    return fused_experts_nvfp4_vec(
        x, *(banks[r] if r in banks else banks[legacy] for r, legacy in roles),
        w, ids, activation, act_fn, bufs=bufs,
    )


_CALLS = {"nvfp4": _nvfp4_call}


def _activation_for(name: str, device: torch.device):
    """The activation this device can actually run, kernel first and torch if it cannot.

    Triton has no target for every device a build has code for -- it refuses gfx90c outright
    ("unsupported target") while the rest of the expert path compiles and runs there. Losing
    a fused activation costs speed; refusing the device costs the device. Probed once, here,
    rather than at the first step, so a failure to compile is a line in the log at start-up
    and not a step that dies.
    """
    from freetoken.layers import activation_torch
    from freetoken.moe.fused_q4_0 import _ACT

    torch_fn = activation_torch.BY_NAME.get(name)
    probe_fn = _ACT.get(name) or next(iter(_ACT.values()), None)
    if probe_fn is None:
        return None
    try:
        with torch.cuda.device(device):
            probe = torch.zeros(1, 2, dtype=torch.bfloat16, device=device)
            probe_fn(probe)
            torch.cuda.synchronize(device)
        return None  # the fused kernel compiles here; let the expert kernel use it
    except Exception as exc:
        if torch_fn is None:
            raise
        logger.info_rank0(
            f"MoE on device {device.index}: activation {name!r} falls back to torch "
            f"({type(exc).__name__}); the expert kernel itself is unaffected"
        )
        return torch_fn


def _pinned(shape, dtype) -> torch.Tensor:
    """A pinned host tensor both devices can address."""
    return torch.zeros(shape, dtype=dtype, device="cpu").pin_memory()


class DeviceMoeExecutor:
    """One other device of this process, serving whichever layers it is offered.

    ``bank_sources`` is the offload cache's own ``{role: [tensor per layer]}``: the host
    side, not the slot cache. Nothing here copies a bank -- a layer's device view is built
    from the pointer the pinned allocation already has, so a layer costs an address.
    """

    def __init__(
        self,
        device_index: int,
        bank_sources: dict[str, list[torch.Tensor]],
        *,
        quant_format: str,
        hidden_size: int,
        top_k: int,
        max_batch: int = 1,
        activation: str = "silu",
        act_fn=None,
        serves_layers=None,
    ) -> None:
        if quant_format not in _CALLS:
            raise NotImplementedError(
                f"in-process device executor has no kernel for {quant_format!r} experts; "
                f"known: {sorted(_CALLS)}"
            )
        self.device_index = int(device_index)
        self.device = torch.device("cuda", self.device_index)
        self._sources = bank_sources
        self._call = _CALLS[quant_format]
        self._activation = activation
        self._act_fn = act_fn if act_fn is not None else _activation_for(activation, self.device)
        self._offered = None if serves_layers is None else frozenset(serves_layers)
        any_role = next(iter(bank_sources))
        self.layers = frozenset(range(len(bank_sources[any_role])))
        self._views: dict[int, dict] = {}
        self._slots: dict[tuple[int, int], int] = {}
        self._next_slot = 0
        self.hits = 0
        self.misses = 0
        self.self_timed = True  # the engine's host clock must not speak for this device

        h, k, b = int(hidden_size), int(top_k), max(1, int(max_batch))
        self._top_k = k
        # The hand-over, in memory both devices reach, at addresses that do not move: a
        # captured graph records them and every replay must find the same ones.
        self._x = _pinned((b, h), torch.bfloat16)
        self._ids = _pinned((b, k), torch.int32)
        self._w = _pinned((b, k), torch.float32)
        # One answer region per slot, not one shared by every layer. The engine reads a
        # slot's answer some time after the doorbell that announced it, and a single
        # buffer makes that read a race with the next layer's write -- the two are only
        # ordered through flags that speak about the slot, not about the buffer.
        self._y = _pinned((_SLOTS, b, h), torch.bfloat16)
        # ready: the engine has published a request in this slot. done: its answer is in _y.
        # slot_layer / slot_bs say what the slot means; both are fixed the first time a slot
        # is handed out, so a replay -- which runs no Python -- still finds them right.
        self._ready = _pinned((_SLOTS,), torch.int64)
        self._done = _pinned((_SLOTS,), torch.int64)
        self._slot_layer = torch.zeros(_SLOTS, dtype=torch.int64)
        self._slot_bs = torch.zeros(_SLOTS, dtype=torch.int64)
        self._near_index = torch.cuda.current_device()
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
        self._far = {
            name: self._view(t)
            for name, t in (("x", self._x), ("ids", self._ids), ("w", self._w))
        }
        self._far_y = self._view(self._y)
        self._near_device = torch.device("cuda", self._near_index)
        # This device's own alias of the answer buffer, for reading it back with a kernel
        # rather than a transfer: a copy engine is a different agent from the one the far
        # device's release fence orders, and a replay showed it reading the buffer before
        # those writes were visible.
        self._near_y = self._view(self._y, index=self._near_index)
        # Diagnostic: a buffer the far device never writes, read through the same path.
        self._const = _pinned((b, h), torch.bfloat16)
        self._const.fill_(0.001)
        self._near_const = self._view(self._const, index=self._near_index)
        self._bufs = self._alloc_bufs(b, k)

        # What this device is worth, measured by this device. The engine times a helper
        # with a host clock around submit and sync, and that span contains the whole step
        # -- this device's own fetch, its GEMM, every earlier helper's join. A helper that
        # overlaps perfectly is therefore charged the makespan, reads back an order of
        # magnitude slow, and is given proportionally less work, which lengthens the step,
        # which confirms the reading. Stream events on this device's own stream measure
        # the compute and nothing else, and a replay does not stop them: this thread runs
        # every step whether or not there is any Python in it.
        with torch.cuda.device(self.device):
            # Three, so the two halves of a slot's life can be told apart: _ev_pre is
            # recorded before the wait, _ev_start after it. pre->start is this device
            # standing idle because the engine has not rung yet; start->end is the work.
            # Which of the two dominates says who is waiting for whom, and that is the
            # only question worth asking about a split that costs more than it saves.
            self._ev_pre = [torch.cuda.Event(enable_timing=True) for _ in range(_SLOTS)]
            self._ev_start = [torch.cuda.Event(enable_timing=True) for _ in range(_SLOTS)]
            self._ev_end = [torch.cuda.Event(enable_timing=True) for _ in range(_SLOTS)]
        # The route count belongs to the step that was timed, so it is taken on the stream
        # (where it is ordered against the compute) rather than from the live buffer, which
        # the engine has already overwritten by the time the answer is harvested.
        self._ids_seen = _pinned((_SLOTS, b, k), torch.int32)
        self._ids_seen_far = self._view(self._ids_seen)
        # A diagnostic, off unless FREETOKEN_DEVICE_TRACE is set: what this device was
        # handed and what it produced, staged on its own stream so the answer is the one
        # the work actually saw rather than whatever the engine has written since.
        self._trace = os.environ.get("FREETOKEN_DEVICE_TRACE", "") == "1"
        self._zero_out = os.environ.get("FREETOKEN_DEVICE_ZERO_OUT", "") == "1"
        self._read_dma = os.environ.get("FREETOKEN_DEVICE_READ_DMA", "") == "1"
        self._const_out = os.environ.get("FREETOKEN_DEVICE_CONST_OUT", "") == "1"
        # The far device raises "done" with a kernel by default; FREETOKEN_DEVICE_HOST_ACK=1
        # raises it from this process's CPU after the stream drains instead. Both were
        # checked in isolation and both are correct; the host form costs a synchronisation
        # per layer and its last run under a graph ended with the server gone.
        self._host_ack = os.environ.get("FREETOKEN_DEVICE_HOST_ACK", "") == "1"
        # Diagnostic: the far device writes a constant instead of computing, so the path by
        # which its answer reaches the engine is exercised with the work taken out.
        self._far_const = os.environ.get("FREETOKEN_DEVICE_FAR_CONST", "") == "1"
        if self._trace:
            self._x_seen = _pinned((_SLOTS, b, h), torch.bfloat16)
            self._y_seen = _pinned((_SLOTS, b, h), torch.bfloat16)
            self._w_seen = _pinned((_SLOTS, b, k), torch.float32)
            self._x_seen_far = self._view(self._x_seen)
            self._y_seen_far = self._view(self._y_seen)
            self._w_seen_far = self._view(self._w_seen)
            self._traced = 0
            # The engine's side of the same slot, staged by nodes of the engine's own graph:
            # what it handed over (x, ids, w) and what it took back (y). Compared with what
            # this device read and wrote only when the scheduler is idle, because only then
            # are both streams known to be drained -- an earlier trace compared too soon and
            # reported zeros that were never there.
            self._x_eng = _pinned((_SLOTS, b, h), torch.bfloat16)
            self._ids_eng = _pinned((_SLOTS, b, k), torch.int32)
            self._w_eng = _pinned((_SLOTS, b, k), torch.float32)
            self._y_eng = _pinned((_SLOTS, b, h), torch.bfloat16)
            self._engine_seen = self._y_eng
        else:
            self._engine_seen = None
        self._checked = 0 if self._trace else 99
        self._retained: dict[int, tuple] = {}
        self._inflight: list[tuple[int, int]] = []
        self._samples: list[tuple[int, float]] = []
        self._samples_lock = threading.Lock()
        # Cumulative, for the report; the samples above are drained by the rate tracker.
        self._seen = 0
        self._busy_s = 0.0
        self._idle_s = 0.0
        self._routes = 0
        self._busy_max = 0.0

        # What the far device is asked to do, and when. A recorded program is a step's
        # worth of slots in the order the engine rings them, kept because a replay runs no
        # Python and so cannot say anything at the time.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._programs: dict[int, list[int]] = {}
        self._recording: list[int] | None = None
        self._pending_bs = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve, name=f"moe-device{self.device_index}", daemon=True
        )
        self._thread.start()

    # --- buffers, allocated once, before any capture can forbid it ----------------------

    def _alloc_bufs(self, max_batch: int, top_k: int) -> dict:
        """Every intermediate the expert path needs, at its largest.

        Sized from the banks themselves so the shapes cannot drift from the weights: the
        gate/up projection is the second dimension of its packed bank, the down projection
        is the second of its own.
        """
        gu = self._sources.get("gate_up") or self._sources["gate_up_packed"]
        dn = self._sources.get("down") or self._sources["down_packed"]
        n2 = int(gu[0].shape[1])
        h = int(dn[0].shape[1])
        routes = max_batch * top_k
        d = n2 // 2
        with torch.cuda.device(self.device):
            return {
                "gate_up": torch.zeros((routes, n2), dtype=torch.bfloat16, device=self.device),
                "inter": torch.zeros((routes, d), dtype=torch.bfloat16, device=self.device),
                "down": torch.zeros((routes, h), dtype=torch.bfloat16, device=self.device),
                "s0": torch.zeros((routes, d), dtype=torch.float32, device=self.device),
                "s1": torch.zeros((routes, d), dtype=torch.float32, device=self.device),
                "wbf": torch.zeros((routes, 1), dtype=torch.bfloat16, device=self.device),
                # Which routes are this executor's, for clearing the rest -- see _into_bufs.
                "valid": torch.zeros((routes, 1), dtype=torch.bool, device=self.device),
                "invalid": torch.zeros((routes, 1), dtype=torch.bool, device=self.device),
            }

    # --- what this device can take ------------------------------------------------------

    def serves(self, layer_id: int) -> bool:
        if self._offered is not None and layer_id not in self._offered:
            return False
        return layer_id in self.layers

    def slot_stats(self) -> dict:
        """No slot cache to report: every expert is reachable, so a route is never a miss."""
        return {"slots": 0, "hits": self.hits, "misses": self.misses}

    # --- host memory, as a device sees it -----------------------------------------------

    def _view(self, host: torch.Tensor, index: int | None = None) -> torch.Tensor:
        """``host`` as a tensor on one device, without a copy and without owning it.

        The alias is asked for with that device current: under UVA the host VA is what every
        device dereferences, but the runtime records which device a registration was made
        for, and ATen checks a blob's pointer against the device it is given -- which is why
        tensor_from_device_ptr is told the device rather than left to infer it.
        """
        from freetoken.kernel.pinned import _load_pinned_extension, tensor_from_device_ptr

        index = self.device_index if index is None else int(index)
        ext = _load_pinned_extension()
        with torch.cuda.device(index):
            addr = ext.host_device_ptr(host.data_ptr())
            return tensor_from_device_ptr(addr, host.shape, host.dtype, index)

    def _views_for(self, layer_id: int) -> dict:
        """This layer's banks as tensors on this device, built once and kept."""
        views = self._views.get(layer_id)
        if views is None:
            views = {
                role: self._view(per_layer[layer_id])
                for role, per_layer in self._sources.items()
            }
            self._views[layer_id] = views
        return views

    def _slot_for(self, layer_id: int, bs: int) -> int:
        key = (int(layer_id), int(bs))
        slot = self._slots.get(key)
        if slot is None:
            if self._next_slot >= _SLOTS:
                raise RuntimeError(
                    f"in-process device executor ran out of handshake slots ({_SLOTS}): "
                    f"more (layer, batch) shapes than it was built for"
                )
            slot = self._slots[key] = self._next_slot
            self._next_slot += 1
            self._slot_layer[slot] = int(layer_id)
            self._slot_bs[slot] = int(bs)
        return slot

    # --- the far device's own thread ----------------------------------------------------

    def _serve(self) -> None:
        """Enqueue work for each slot the engine says it will ring, then go back to sleep.

        The thread decides WHAT runs; the flags decide WHEN. Everything it puts on the
        stream is asynchronous, including the wait -- so a step's whole chain goes in at
        once and the thread is free again long before the device has started on it. That is
        what keeps this off the critical path: the engine's stream and this one are only
        ever coupled through two words in host memory.
        """
        from freetoken.kernel import handshake

        torch.cuda.set_device(self.device)
        ready_ptr, done_ptr = self._ready.data_ptr(), self._done.data_ptr()
        # The engine runs under inference mode and everything here was allocated inside
        # it, which makes these inference tensors: writing to one from a thread that is
        # not itself in inference mode is refused outright. The mode is thread-local, so
        # this thread enters it too -- for the same reason the engine did.
        with torch.inference_mode():
            while True:
                item = self._queue.get()
                if item is None:
                    return
                slots = item if isinstance(item, list) else (item,)
                acks: list[tuple[int, int]] = []
                try:
                    self._harvest()
                    with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                        for slot in slots:
                            bs = int(self._slot_bs[slot])
                            self._ev_pre[slot].record(self._stream)
                            handshake.wait(ready_ptr, slot)
                            # Between the wait and the doorbell is this device's work and
                            # only this device's work; the events bracket exactly that.
                            self._ev_start[slot].record(self._stream)
                            self._ids_seen_far[slot][:bs].copy_(self._far["ids"][:bs])
                            if self._trace:
                                self._x_seen_far[slot][:bs].copy_(self._far["x"][:bs])
                                self._w_seen_far[slot][:bs].copy_(self._far["w"][:bs])
                            self._compute(int(self._slot_layer[slot]), bs, slot)
                            if self._trace:
                                self._y_seen_far[slot][:bs].copy_(self._far_y[slot][:bs])
                                self._check(int(self._slot_layer[slot]), bs, slot)
                            self._ev_end[slot].record(self._stream)
                            if self._host_ack:
                                acks.append((slot, bs))
                            else:
                                handshake.doorbell(ready_ptr, done_ptr, slot)
                                self._inflight.append((slot, bs))
                    # Host acknowledgement, when asked for (see _host_ack). An earlier note
                    # here claimed the engine read zeros where this device had written a
                    # partial; that reading came from an unsynchronised trace and was not
                    # evidence. Isolated, both acknowledgements carry this device's writes
                    # to the engine intact under replay.
                    for slot, bs in acks:
                        self._ev_end[slot].synchronize()
                        self._ready[slot] = 0
                        self._done[slot] = 1
                        self._inflight.append((slot, bs))
                except BaseException as exc:  # noqa: BLE001 -- reported, then released
                    self._fail(exc, slots)
                    return

    def _fail(self, exc: BaseException, slots) -> None:
        """Release everyone waiting on us, so a failure is reported rather than a hang.

        The engine's side of this handshake is a device-side spin on a word only this
        thread raises. Dying quietly would not fail the engine, it would stop it -- no
        message, nothing in the log, a stream that never drains. So the flags are raised by
        hand and the reason is kept for the next submit to tell.
        """
        self._error = exc
        logger.error(
            f"MoE on device {self.device_index} failed and is being taken out of the "
            f"split: {type(exc).__name__}: {exc}"
        )
        with torch.inference_mode():
            self._done.fill_(1)

    def _harvest(self) -> None:
        """Turn last step's events into samples, for any pair that has actually landed.

        Asked, never waited on: an event that has not finished is left in place for the
        next pass. A sample only says something about this device if the work it brackets
        is over, and stalling here to make sure of it would put the host back on the
        critical path the whole arrangement exists to keep it off.
        """
        if not self._inflight:
            return
        still: list[tuple[int, int]] = []
        fresh: list[tuple[int, float]] = []
        for slot, bs in self._inflight:
            if not self._ev_end[slot].query():
                still.append((slot, bs))
                continue
            seconds = self._ev_start[slot].elapsed_time(self._ev_end[slot]) / 1e3
            idle = self._ev_pre[slot].elapsed_time(self._ev_start[slot]) / 1e3
            routes = int((self._ids_seen[slot][:bs] >= 0).sum())
            self._seen += 1
            self._busy_s += seconds
            self._idle_s += max(0.0, idle)
            self._routes += routes
            self._busy_max = max(self._busy_max, seconds)
            if self._trace and self._traced < 12:
                self._traced += 1
                ids = self._ids_seen[slot][:bs]
                x, y = self._x_seen[slot][:bs].float(), self._y_seen[slot][:bs].float()
                w = self._w_seen[slot][:bs]
                logger.info_rank0(
                    f"  trace slot {slot:2d} layer {int(self._slot_layer[slot]):2d}: "
                    f"routes={int((ids >= 0).sum())} ids={ids.reshape(-1).tolist()} "
                    f"wsum={float(w.sum()):.4f} wmax={float(w.abs().max()):.4f} "
                    f"|x|={float(x.abs().sum()):.4f} xmax={float(x.abs().max()):.4f} "
                    f"|y|={float(y.abs().sum()):.4f} ymax={float(y.abs().max()):.4f}"
                )
            if routes > 0 and seconds > 0.0:
                fresh.append((routes, seconds))
        self._inflight = still
        if fresh:
            with self._samples_lock:
                self._samples.extend(fresh)

    def timing_summary(self) -> str | None:
        """What this device did with the layers it was given, per layer, in words.

        ``idle`` is time spent waiting for the engine to ring; ``busy`` is the expert
        work. Idle much larger than busy means the engine sets the pace and this device
        is free -- give it more. Idle near zero means the opposite: this device is always
        behind, and the engine's stream is the one doing the waiting.
        """
        if not self._seen:
            return None
        n = self._seen
        return (
            f"MoE device {self.device_index}: {n} layer-steps measured, "
            f"busy {self._busy_s / n * 1e3:.3f} ms/layer (max {self._busy_max * 1e3:.3f}), "
            f"idle {self._idle_s / n * 1e3:.3f} ms/layer, "
            f"{self._routes / n:.2f} routes/layer, "
            f"busy share {100 * self._busy_s / max(1e-9, self._busy_s + self._idle_s):.1f}%"
        ) + self._idle_comparison()

    def _idle_comparison(self) -> str:
        """At idle, both streams drained: does this device see what the engine sent, and does
        the engine get back what this device wrote? Per slot, for the last step served."""
        if not self._trace or not self._next_slot:
            return ""
        n = self._next_slot

        def differs(a, b):
            a, b = a[:n].float(), b[:n].float()
            both_nan = torch.isnan(a) & torch.isnan(b)
            return ((a != b) & ~both_nan).flatten(1).any(dim=1)

        dx = differs(self._x_eng, self._x_seen)
        di = differs(self._ids_eng, self._ids_seen)
        dw = differs(self._w_eng, self._w_seen)
        dy = differs(self._y_eng, self._y_seen)
        nan_eng = torch.isnan(self._y_eng[:n].float()).flatten(1).any(dim=1)
        nan_far = torch.isnan(self._y_seen[:n].float()).flatten(1).any(dim=1)

        def first(mask):
            idx = mask.nonzero()
            return int(idx[0]) if idx.numel() else -1

        return (
            f" | idle palyginimas {n} lizdų: x skiriasi {int(dx.sum())} (pirmas {first(dx)}), "
            f"ids {int(di.sum())} (pirmas {first(di)}), w {int(dw.sum())} (pirmas {first(dw)}), "
            f"y skiriasi {int(dy.sum())} (pirmas {first(dy)}); "
            f"NaN y: variklio {int(nan_eng.sum())}, iGPU {int(nan_far.sum())}"
        )

    def take_samples(self) -> list[tuple[int, float]]:
        """Every (routes, seconds) measured since the last ask, and clear them."""
        with self._samples_lock:
            samples, self._samples = self._samples, []
        return samples

    def _check(self, layer_id: int, bs: int, slot: int) -> None:
        """Recompute this layer the ordinary way and say whether the fast path agrees.

        Diagnostic only. The allocation-free path exists because an allocation is refused
        while a capture is underway; a replay is not a capture, so here -- where the work
        is actually done -- the plain path can still be run for comparison.
        """
        if self._checked >= 8:
            return
        self._checked += 1
        # From the snapshot, not the live buffers: the engine reuses x/ids/w for the next
        # layer the moment this one is acknowledged, so a reference built from what is
        # there now would answer a different question. The snapshot was taken on this
        # stream, immediately before the work, and is exactly what the work read.
        ref = self._call(
            self._views[layer_id], self._x_seen_far[slot][:bs], self._w_seen_far[slot][:bs],
            self._ids_seen_far[slot][:bs], self._activation, self._act_fn, None,
        )
        got = self._y_seen_far[slot][:bs]
        torch.cuda.synchronize(self.device)
        r, g = ref.float(), got.float()
        xs = self._x_seen[slot][:bs].float()
        d = (r - g).abs().max().item()
        scale = max(1e-6, r.abs().max().item())
        logger.info_rank0(
            f"  check layer {layer_id:2d}: routes="
            f"{int((self._ids_seen[slot][:bs] >= 0).sum())} "
            f"x_nan={bool(torch.isnan(xs).any())} |x|max={xs.abs().max().item():.4f} "
            f"|ref|max={scale:.5f} |got|max={g.abs().max().item():.5f} "
            f"skirtumas={d:.5f} ({100 * d / scale:.1f}%)"
        )

    def _compute(self, layer_id: int, bs: int, slot: int) -> None:
        """One layer's share, entirely into buffers this executor already owns."""
        if self._far_const:
            self._far_y[slot][:bs].fill_(0.001)
            return
        bufs = dict(self._bufs)
        bufs["out"] = self._far_y[slot][:bs]
        self._call(
            self._views[layer_id], self._far["x"][:bs], self._far["w"][:bs],
            self._far["ids"][:bs], self._activation, self._act_fn, bufs,
        )

    # --- capture, and the step that replays it ------------------------------------------

    def begin_record(self, bs: int) -> None:
        """Capture is starting: note the slots, do not run them.

        Nothing this device does is in the graph -- it cannot be, a replay of a graph
        spanning both devices faults -- so during capture it must stay still: the engine's
        doorbell is being recorded, not executed, and a device told to wait for a word that
        will not be raised until replay would wait for it now.
        """
        self._recording = []
        self._pending_bs = int(bs)

    def end_record(self) -> None:
        """Capture is over: this is the step this graph will ask for, every replay."""
        if self._recording is not None:
            self._programs[self._pending_bs] = self._recording
            logger.info_rank0(
                f"MoE on device {self.device_index}: {len(self._recording)} layers "
                f"recorded for the bs={self._pending_bs} graph"
            )
        self._recording = None

    def begin_step(self, bs: int) -> None:
        """A replay is about to start: hand the far device the whole step at once."""
        program = self._programs.get(int(bs))
        if program:
            self._queue.put(program)

    # --- the two calls every executor answers -------------------------------------------

    def decode_submit(self, layer_id: int, hidden_states, topk_weights, topk_ids):
        """Publish this layer's share, ring for it, and return without waiting.

        Four things go on the engine's stream and a graph records all four: three copies
        into host memory and the doorbell. Nothing here synchronises, so the caller goes on
        filling its own stream -- which is the point of the split.
        """
        from freetoken.kernel import handshake

        if self._error is not None:
            raise RuntimeError(
                f"MoE on device {self.device_index} is not serving: "
                f"{type(self._error).__name__}: {self._error}"
            ) from self._error

        bs = hidden_states.shape[0]
        slot = self._slot_for(layer_id, bs)
        self._views_for(layer_id)  # an address, taken before a capture can forbid asking

        # Hold the sources for the life of the slot. A capture turns each copy into a
        # node reading a FIXED address, and a dtype cast makes a temporary the allocator
        # reclaims the moment this returns -- after which the same address is handed to
        # something else in the same graph, and every replay copies whatever now lives
        # there. One set per slot, so this is bounded by the number of shapes rather
        # than by the number of steps.
        ids32 = topk_ids.to(torch.int32).contiguous()
        w32 = topk_weights.to(torch.float32).contiguous()
        x_src = hidden_states.contiguous()
        self._retained[slot] = (x_src, ids32, w32)

        # Straight to the pinned host tensors, not through this device's alias of them.
        # These were allocated by torch, so torch knows they are page-locked and issues a
        # real asynchronous device-to-host copy -- the same thing the CPU executor does,
        # and the thing a capture records correctly. Writing through the alias instead
        # makes it a device-to-device copy over a host-mapped pointer, which is what a
        # replay got wrong: eager steps were right and every captured one was rubbish.
        self._x[:bs].copy_(x_src, non_blocking=True)
        self._ids[:bs].copy_(ids32, non_blocking=True)
        self._w[:bs].copy_(w32, non_blocking=True)
        if self._trace:
            self._x_eng[slot][:bs].copy_(x_src, non_blocking=True)
            self._ids_eng[slot][:bs].copy_(ids32, non_blocking=True)
            self._w_eng[slot][:bs].copy_(w32, non_blocking=True)
        handshake.doorbell(self._done.data_ptr(), self._ready.data_ptr(), slot)

        if self._recording is not None:
            self._recording.append(slot)  # a replay will ring this; nobody serves it now
        else:
            self._queue.put(slot)
        self.hits += int(topk_ids.numel())
        return (slot, bs, hidden_states.dtype)

    def decode_sync(self, pending):
        """Hold this stream until the partial is there, then take a copy of it.

        A stream wait, not a host one: the caller is mid-step with its own work queued
        behind this, and a host sync would serialise the two devices the split exists to
        overlap. The copy is needed because the buffer is reused by the next step.
        """
        from freetoken.kernel import handshake

        slot, bs, dtype = pending
        handshake.wait(self._done.data_ptr(), slot)
        # And back the same way: a host-to-device copy out of the pinned tensor, into
        # memory of this device. The copy is needed regardless -- the buffer is reused by
        # the next layer -- so nothing is lost by taking the ordinary path.
        out = torch.empty((bs, self._y.shape[2]), dtype=self._y.dtype, device=self._near_device)
        if self._const_out:  # diagnostic: same read path, content nobody writes
            torch.mul(self._near_const[:bs], 1.0, out=out)
        elif self._zero_out:  # diagnostic: keep the handshake, discard the answer
            out.zero_()
        elif self._read_dma:  # diagnostic: the transfer path
            out.copy_(self._y[slot][:bs], non_blocking=True)
        else:
            # A kernel read, ordered against the wait above like any other kernel.
            torch.mul(self._near_y[slot][:bs], 1.0, out=out)
        if self._engine_seen is not None:
            self._engine_seen[slot][:bs].copy_(out, non_blocking=True)
        return out.to(dtype) if out.dtype != dtype else out

    def shutdown(self) -> None:
        """No process, no files, no registrations of our own -- just the thread."""
        self._queue.put(None)
        self._views.clear()
        self._retained.clear()


__all__ = ["DeviceMoeExecutor"]
