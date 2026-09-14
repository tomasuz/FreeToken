"""Experts computed on a second GPU of this process, reading the engine's own host banks.

The sibling of :class:`~freetoken.moe.worker_executor.WorkerMoeExecutor`, and it exists
because that one should not have to.

A worker is a process, and it is a process for exactly one reason: a device whose runtime
settings differ from this one's can only be driven from somewhere those settings are the
process's own -- an integrated GPU with no code in the build presents as an older
architecture, and everything it touches has to be found or built for that instead. A build
that has code for the device needs none of it, and then the device is just another one this
process owns.

What the process boundary cost is the whole of the worker's machinery: the banks published
as files, mapped a second time by the child, a registration budget spent twice over, a
doorbell rung through shared memory and a device-side spin-wait on a flag. None of it is
needed here. The banks are pinned host memory registered portable, so the device pointer
they already have is valid on every device this process holds: the same tensors the main
device reads are read from the second one, and a step hands over a hidden state and gets
back a partial.

Measured on the machine this was written for, the worker's round trip put the integrated
GPU at 0.33 GB/s end to end -- a number that describes the handshake rather than the
device. This path has no handshake to describe.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _nvfp4_call(banks: dict, x, w, ids, activation, act_fn):
    from freetoken.moe.fused_nvfp4_vec import fused_experts_nvfp4_vec

    roles = (
        ("gate_up", "gate_up_packed"), ("gate_up_scale", "gate_up_scale"),
        ("gate_up_global", "gate_up_global"), ("down", "down_packed"),
        ("down_scale", "down_scale"), ("down_global", "down_global"),
    )
    return fused_experts_nvfp4_vec(
        x, *(banks[r] if r in banks else banks[legacy] for r, legacy in roles),
        w, ids, activation, act_fn,
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


class DeviceMoeExecutor:
    """One other device of this process, serving whichever layers it is offered.

    ``bank_sources`` is the offload cache's own ``{role: [tensor per layer]}``: the host
    side, not the slot cache. Nothing here copies a bank -- a layer's device view is built
    from the pointer the pinned allocation already has, so a layer costs an address and the
    first use of a layer costs nothing but building the views.
    """

    def __init__(
        self,
        device_index: int,
        bank_sources: dict[str, list[torch.Tensor]],
        *,
        quant_format: str,
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
        self._staging: dict = {}
        # Its own stream, so the work overlaps this step's GEMM on the main device rather
        # than queueing behind it. Events, not syncs, join the two back together.
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.Stream(device=self.device)
        self.hits = 0
        self.misses = 0

    # --- what this device can take ------------------------------------------------------

    def serves(self, layer_id: int) -> bool:
        if self._offered is not None and layer_id not in self._offered:
            return False
        return layer_id in self.layers

    def slot_stats(self) -> dict:
        """No slot cache to report: every expert is reachable, so a route is never a miss."""
        return {"slots": 0, "hits": self.hits, "misses": self.misses}

    # --- the banks, as this device sees them --------------------------------------------

    def _views_for(self, layer_id: int) -> dict:
        """This layer's banks as tensors on this device, built once and kept.

        The pages are registered portable, which is what makes this legal: one registration,
        made for the device that allocated them, and every device of the process gets a
        pointer. Were they not, this would be the worker's problem all over again -- a second
        registration of the same pages, refused by the driver as already mapped.
        """
        views = self._views.get(layer_id)
        if views is not None:
            return views
        from freetoken.kernel.pinned import _load_pinned_extension, tensor_from_device_ptr

        ext = _load_pinned_extension()
        views = {}
        # The alias has to be asked for with THIS device current. Under UVA the host VA is
        # what every device dereferences, and `device_ptr` says so by handing back
        # `data_ptr()` -- but the runtime still records which device a registration was made
        # for, and torch checks a blob's pointer against the device it is being given. Ask
        # the runtime for this device's alias instead, in this device's context, and the two
        # agree.
        with torch.cuda.device(self.device):
            for role, per_layer in self._sources.items():
                host = per_layer[layer_id]
                addr = (
                    host.data_ptr() if host.is_cuda
                    else ext.host_device_ptr(host.data_ptr())
                )
                views[role] = tensor_from_device_ptr(
                    addr, host.shape, host.dtype, self.device_index
                )
        self._views[layer_id] = views
        return views

    # --- the two calls every executor answers -------------------------------------------

    def decode_submit(self, layer_id: int, hidden_states, topk_weights, topk_ids):
        """Compute this layer's share on the other device.

        Synchronous, and deliberately so for now. The shape this interface was built for is
        submit-then-sync, so several executors overlap; joining two devices that way needs a
        cross-device event, and waiting on one from the other device's stream crashes inside
        the runtime on this stack (THCPEvent_wait -> hipStreamCreateWithPriority). Overlap is
        a speed question and this is a correctness one, so the join is a stream sync until
        the event path is understood -- the split still decides how much comes here, and the
        rate tracker still measures what it cost, including the sync.
        """
        views = self._views_for(layer_id)
        # Through the host, not device to device. A direct cross-device copy records an
        # event on the other device's null stream inside the runtime, and that path aborts
        # on this ROCm build (dispatch_to -> hipEventRecord -> Stream::terminate). Staging
        # costs two ordinary transfers of an activation -- one hidden vector, its ids and
        # its weights -- which is what the worker moved too. The banks are what must not be
        # copied, and they are not: they are read where they lie.
        torch.cuda.current_stream().synchronize()
        x = self._via_host(hidden_states)
        ids = self._via_host(topk_ids)
        w = self._via_host(topk_weights)
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            y = self._call(views, x, w, ids, self._activation, self._act_fn)
            self._stream.synchronize()
        self.hits += int((topk_ids >= 0).sum()) if topk_ids.numel() else 0
        return (y, hidden_states.device, hidden_states.dtype)

    def decode_sync(self, pending):
        """Bring the partial back, the same way it went out."""
        y, dst_device, dtype = pending
        host = y.to("cpu")
        out = host.to(dst_device)
        return out.to(dtype) if out.dtype != dtype else out

    def _via_host(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` on this executor's device, staged through pinned host memory.

        The buffer is kept per (shape, dtype) because a decode step has one shape and asks
        for it again every token; allocating a pinned buffer per step would cost more than
        the transfer.
        """
        key = (tuple(t.shape), t.dtype)
        buf = self._staging.get(key)
        if buf is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device="cpu").pin_memory()
            self._staging[key] = buf
        buf.copy_(t)
        with torch.cuda.device(self.device):
            return buf.to(self.device)

    def shutdown(self) -> None:
        """Nothing to tear down: no process, no files, no registrations of our own."""
        self._views.clear()


__all__ = ["DeviceMoeExecutor"]
