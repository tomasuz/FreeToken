"""MoE experts computed by a worker process bound to another accelerator.

The sibling of :class:`~freetoken.moe.cpu_executor.CpuMoeExecutor`, and it exists for the
same reason with a different destination: some layers are cheaper to compute where their
weights already are than to stream to the main device. The CPU executor sends them to the
host's cores; this one sends them to an accelerator the *engine's own process cannot
drive*.

That last part is the whole point. Which architecture a runtime targets, and which library
it loads, are process-wide settings latched on first use, so a machine holding devices that
disagree about them cannot serve them all from one process -- honouring one breaks the
other. A worker process gets its own environment, and the disagreement stops being a
contradiction. Nothing here names a vendor, an architecture, or a kind of device: the
caller supplies a device index and whatever environment that device needs.

The parent's side of a step never touches the worker's device. It writes activations into
shared host memory, rings a doorbell, waits on a flag, and reads the result back -- exactly
the shape the CPU executor already had, which is why the layer above does not change.

This is the correctness-first cut: the handoff is a polled flag and every step
synchronises. The CPU executor's stream memory operations (submit and wait executed on the
GPU front end, so a captured graph rides the handshake) are the performance follow-up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import torch

from freetoken.moe.shared_host import create_shared
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Legacy control words, still used for shutdown and for reporting cache statistics.
_STOP, _HITS, _MISSES, _UP = 0, 1, 2, 3
_CTL_SLOTS = 4

# Flag slots per layer: one per distinct decode batch size that gets its own captured
# graph. The slot is what makes the handshake capturable -- a replay must write the same
# address every time, so the layer and batch size have to be baked into the address rather
# than written into a shared word each step. More sizes than this keeps the polled path,
# which is functional and only slower.
_SLOTS_PER_LAYER = 8

# A cold worker pays process start, torch import, and a JIT compile of the expert kernel
# for its architecture. The compile is the long pole and happens once per machine.
_START_TIMEOUT_S = 900
_STEP_TIMEOUT_S = 120

_DTYPE_NAMES = {
    torch.uint8: "uint8",
    torch.int32: "int32",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.int64: "int64",
}


def _visibility_var() -> str:
    return "HIP_VISIBLE_DEVICES" if getattr(torch.version, "hip", None) else "CUDA_VISIBLE_DEVICES"


class SharedBankCatalogue:
    """Where every offloaded layer's expert weights live, by name and layer.

    A worker used to be handed a copy of one layer, which is why it could only ever serve
    that layer: it had no way to reach any other. The banks are named files now, so what a
    worker needs is not weights but paths -- it maps the same pages the engine already
    holds, for as many layers as it is asked about, and nothing is duplicated.

    Paths are derived from the tag the loader shared its banks under rather than carried
    out of the loader, because the names are deterministic and the objects are not: the
    loader hands back tensors, and threading bank handles through it to reach a filename
    would be a worse coupling than recomputing the filename.

    A layer whose file is absent is simply not listed. That is the resident case -- its
    bank was uploaded to VRAM and released -- and such a layer never misses, so there is
    nothing for a worker to serve.
    """

    def __init__(
        self,
        tag: str,
        sources: dict[str, list[torch.Tensor]],
        num_experts: int,
    ) -> None:
        from freetoken.moe.host_banks import shared_bank_name

        self.num_experts = num_experts
        self._paths: dict[str, dict[int, str]] = {}
        self._shapes: dict[str, list[int]] = {}
        self._dtypes: dict[str, str] = {}
        from freetoken.moe.shared_host import shm_dir

        for name, per_layer in sources.items():
            found = {}
            for layer_id, tensor in enumerate(per_layer):
                path = os.path.join(shm_dir(), shared_bank_name(tag, name, layer_id))
                if os.path.exists(path):
                    found[layer_id] = path
            self._paths[name] = found
            self._shapes[name] = list(per_layer[0].shape)
            self._dtypes[name] = _DTYPE_NAMES[per_layer[0].dtype]
        names = list(self._paths)
        self.layers = sorted(
            set.intersection(*(set(self._paths[n]) for n in names)) if names else set()
        )

    def spec(self) -> dict:
        return {
            name: {
                "paths": {str(layer): self._paths[name][layer] for layer in self.layers},
                "shape": self._shapes[name],
                "dtype": self._dtypes[name],
            }
            for name in self._paths
        }


class WorkerMoeExecutor:
    """Serves one MoE layer's experts from a worker process on ``device_index``.

    ``env`` is applied to the child before it starts, and is how a device that needs a
    different runtime configuration than the parent's gets it. ``banks`` are the layer's
    packed expert weights; they are copied once into shared memory at construction, after
    which both processes address the same pages.

    ``activation_backend`` picks the activation implementation in the worker: ``"auto"``
    uses the compiled kernel where it builds and torch where it does not, which is what
    keeps a device usable when the kernel compiler has no backend for it.
    """

    def __init__(
        self,
        device_index: int,
        banks: dict[str, torch.Tensor],
        *,
        ggml_type: int,
        activation: str = "silu",
        max_batch: int = 64,
        hidden_size: int | None = None,
        top_k: int | None = None,
        env: dict[str, str] | None = None,
        activation_backend: str = "auto",
        slots: int | None = None,
    ) -> None:
        self.device_index = device_index
        self._proc: subprocess.Popen | None = None
        self._bufs: list = []
        self._keep_spec = False
        self._log = None
        self._log_path = ""

        h = hidden_size
        k = top_k if top_k is not None else 1
        tag = uuid.uuid4().hex[:12]

        def share(name: str, shape, dtype):
            buf = create_shared(f"freetoken-{tag}-{name}", tuple(shape), dtype)
            self._bufs.append(buf)
            return buf

        self._ctl = share("ctl", (_CTL_SLOTS,), torch.int64)
        self._ctl.tensor.zero_()
        num_layers = (max(banks.layers) + 1) if banks.layers else 1
        self._capacity = num_layers * _SLOTS_PER_LAYER
        # ready/done are the handshake the GPU front end drives; the descriptors tell the
        # worker what a raised slot means, written once when the slot is handed out.
        self._flags = {
            "ready": share("ready", (self._capacity,), torch.int64),
            "done": share("done", (self._capacity,), torch.int64),
            "slot_layer": share("slot_layer", (self._capacity,), torch.int64),
            "slot_bs": share("slot_bs", (self._capacity,), torch.int64),
        }
        for buf in self._flags.values():
            buf.tensor.zero_()
        self._io = {
            "x": share("x", (max_batch, h), torch.bfloat16),
            "ids": share("ids", (max_batch, k), torch.int32),
            "w": share("w", (max_batch, k), torch.float32),
            "y": share("y", (max_batch, h), torch.bfloat16),
        }
        self._slots: dict[tuple[int, int], int] = {}
        self._next_slot = 0
        # Whether this device can compute an expert where it already lies, instead of
        # having a copy of it made in the device's own memory. Decided by measurement in
        # the child, which is the only process that can ask its own device; the parent
        # only carries the answer. Default is to try, since an expert not copied is an
        # expert not paid for twice.
        self.reads_in_place = os.getenv("FREETOKEN_WORKER_READ_IN_PLACE", "1") == "1"
        num_experts = banks.num_experts
        self.layers = sorted(banks.layers)
        # One token's experts are read by a single launch, so top_k is the floor. A wider
        # step is split into several launches by the worker's cache, which is what keeps
        # this a memory choice rather than a cap on batch width. Unset holds a layer's whole
        # expert count, which never evicts.
        self.slots = min(num_experts, max(k, int(slots or num_experts)))

        spec = {
            "control": self._entry(self._ctl),
            "banks": banks.spec(),
            "io": {n: self._entry(b) for n, b in self._io.items()},
            "flags": {n: self._entry(b) for n, b in self._flags.items()},
            "slots": self.slots,
            "read_in_place": self.reads_in_place,
            "ggml_type": int(ggml_type),
            "activation": activation,
            # "auto" lets the worker fall back to torch where the compiled activation has
            # no backend for its device; "kernel"/"torch" force one.
            "activation_backend": activation_backend,
        }
        self._spec_path = os.path.join(tempfile.gettempdir(), f"freetoken-worker-{tag}.json")
        with open(self._spec_path, "w") as fh:
            json.dump(spec, fh)

        self._spawn(env or {})

    @staticmethod
    def _entry(buf) -> dict:
        return {
            "path": buf.path,
            "shape": list(buf.tensor.shape),
            "dtype": _DTYPE_NAMES[buf.tensor.dtype],
        }

    def _spawn(self, env: dict[str, str]) -> None:
        child_env = dict(os.environ)
        # The child has to import the same freetoken the parent is running, and the parent
        # may have been found through sys.path rather than an installed package (a source
        # checkout, an editable install, a test run). sys.path is not inherited across a
        # spawn, so hand the package's own root over explicitly.
        import freetoken

        # The directory *containing* the package, which is what an import needs on the
        # path -- derived from the package itself rather than by counting parents up from
        # this file, so moving this module does not silently break the child.
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(freetoken.__file__)))
        existing = child_env.get("PYTHONPATH", "")
        if pkg_root not in existing.split(os.pathsep):
            child_env["PYTHONPATH"] = (
                pkg_root + (os.pathsep + existing if existing else "")
            )
        # Drop an inherited kernel-architecture selection. This process picks one from the
        # devices *it* can see and exports it; inherited into a child pinned to a different
        # device, it names architectures that child is not running on, and a fat binary with
        # no code for the actual device faults on the first launch rather than failing to
        # load. The child derives its own once the visibility below applies.
        child_env.pop("PYTORCH_ROCM_ARCH", None)
        # Restrict the child to its device *before* its overrides, so an override that
        # names visibility itself still wins -- the caller may know better than we do.
        child_env[_visibility_var()] = str(self.device_index)
        child_env.update(env)
        # Files, not pipes. A kernel compiler that fails can print megabytes -- Triton
        # dumps a full MLIR reproducer -- and a child writing to a pipe nobody is draining
        # blocks in write() once the buffer fills. The parent is meanwhile polling a flag
        # that will now never be set, so the whole thing hangs on the one path where the
        # output actually matters.
        self._log_path = self._spec_path.replace(".json", ".log")
        self._log = open(self._log_path, "w+")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "freetoken.moe._worker_main", self._spec_path],
            env=child_env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            self._await_flag(_UP, 1, _START_TIMEOUT_S, "start")
        except Exception:
            # The spec names every shared buffer; keeping it lets the same child be
            # started again by hand, which is the only way to debug a crash this side
            # only sees as a signal number.
            logger.warning_rank0(
                f"MoE worker spec kept for inspection: {self._spec_path} "
                f"(child output: {self._log_path})"
            )
            self._keep_spec = True
            raise
        logger.info_rank0(f"MoE worker up on device {self.device_index}")
        self._enable_stream_handshake()

    def _await_flag(self, slot: int, want: int, timeout: float, what: str) -> None:
        """Poll a control flag, failing loudly if the worker died or stopped answering.

        A dead child is the likely outcome the first time a device is tried, so the wait
        checks liveness rather than only the clock: the child's own error is far more
        useful than "timed out"."""
        deadline = time.monotonic() + timeout
        flags = self._ctl.tensor
        while int(flags[slot]) != want:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"MoE worker for device {self.device_index} exited with {rc} during "
                    f"{what}: {self._child_error()}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"MoE worker for device {self.device_index} did not answer within "
                    f"{timeout:.0f}s during {what}"
                )
            time.sleep(0.001)

    def _child_error(self) -> str:
        """The child's output, trimmed from both ends.

        A crash inside a kernel compiler buries the useful lines under pages of dumped IR,
        so a plain tail of the stream is the one part guaranteed not to say what happened.
        Keep the start (where the traceback begins) and the end (where it names the error),
        and say how much was dropped between them.
        """
        try:
            self._log.flush()
            with open(self._log_path) as fh:
                text = fh.read().strip()
        except OSError:
            text = ""
        if not text:
            return "(no output)"
        lines = text.splitlines()
        if len(lines) <= 40:
            return "\n" + text
        head, tail = lines[:15], lines[-15:]
        return "\n" + "\n".join(
            head + [f"    ... {len(lines) - 30} lines omitted ..."] + tail
        )

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the worker's device. Returns a tensor on the caller's.

        Mirrors ``CpuMoeExecutor.decode``'s contract; the submit/sync split it has for
        overlap comes with the stream-memop handshake, not before it.
        """
        bs = hidden_states.shape[0]
        assert bs >= 1, "batch size doubles as the doorbell, so it cannot be zero"
        assert bs <= self._io["x"].tensor.shape[0], (
            f"batch {bs} exceeds the worker's max_batch {self._io['x'].tensor.shape[0]}"
        )
        pending = self.decode_submit(layer_id, hidden_states, topk_weights, topk_ids)
        return self.decode_sync(pending)

    def _enable_stream_handshake(self) -> None:
        """Try to move the handshake onto the GPU front end, and say what happened.

        The polled version cannot be captured: the wait is Python, so a graph would record
        the copies around a worker that never ran and replay stale output. Stream memory
        operations put both halves -- raise ready, wait on done -- on the stream itself,
        where a capture records them like any other node and a replay drives the worker for
        real. That is what the CPU executor already does; the mechanism is not specific to
        where the work lands.

        Everything the GPU touches has to be registered for it to address: the flags it
        writes and waits on, and the activation buffers it copies through. Registering a
        shared mapping is what makes those pages reachable from both the device and the
        other process at once.
        """
        self.stream_handshake = False
        if not torch.cuda.is_available():
            return
        if os.getenv("FREETOKEN_WORKER_STREAM_HANDSHAKE", "0") != "1":
            # Off by default, and deliberately. The handshake below is the thing that makes
            # a captured decode possible, and it does capture -- but the worker's own
            # contribution is not yet right under replay, so capture is disabled anyway and
            # this path buys nothing while it lasts. It also carries a fault of its own:
            # exercised here, the address-issued copies produce an illegal access that
            # surfaces asynchronously a step later. Both are the same piece of unfinished
            # work; whoever picks it up can turn this on and see them.
            return
        try:
            from freetoken.kernel import _cpu_moe
        except Exception as exc:  # the extension is optional; the polled path still works
            logger.info_rank0(
                f"worker on device {self.device_index}: stream handshake unavailable "
                f"({type(exc).__name__}), using the polled one -- decode will not be "
                f"captured into CUDA graphs"
            )
            return
        from freetoken.kernel.pinned import alloc_pinned_tensor

        probe = alloc_pinned_tensor(1, dtype=torch.int64)
        probe.zero_()
        if not _cpu_moe.memops_probe(
            torch.cuda.current_stream().cuda_stream, probe.data_ptr()
        ):
            logger.info_rank0(
                f"worker on device {self.device_index}: CUDA stream memory operations are "
                f"not supported here, using the polled handshake -- decode will not be "
                f"captured into CUDA graphs"
            )
            return
        for buf in list(self._flags.values()) + list(self._io.values()):
            buf.pin()
        self._memops = _cpu_moe
        self.stream_handshake = True

    def _slot_for(self, layer_id: int, bs: int) -> int | None:
        """The flag slot for this (layer, batch size), or ``None`` if none is left.

        Handed out on first use and never moved, because a captured graph bakes the address
        in. The descriptors are written here, before any capture can happen, so the worker
        can tell what a raised slot means without anything being written per step.
        """
        key = (int(layer_id), int(bs))
        slot = self._slots.get(key)
        if slot is not None:
            return slot
        if self._next_slot >= self._capacity:
            return None  # more shapes than slots: fall back rather than reuse one
        slot = self._next_slot
        self._next_slot += 1
        self._slots[key] = slot
        self._flags["slot_layer"].tensor[slot] = layer_id
        self._flags["slot_bs"].tensor[slot] = bs
        return slot

    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """Ring the doorbell and return without waiting.

        Splitting the step this way is what lets several executors run at once. A placement
        that hands work to three devices and then waits for each in turn would take the sum
        of their times, not the longest -- which is the opposite of the arrangement it was
        computed to produce.
        """
        bs = hidden_states.shape[0]
        assert bs >= 1, "batch size doubles as the doorbell, so it cannot be zero"
        assert bs <= self._io["x"].tensor.shape[0], (
            f"batch {bs} exceeds the worker's max_batch {self._io['x'].tensor.shape[0]}"
        )
        slot = self._slot_for(layer_id, bs) if self.stream_handshake else None
        ids32 = topk_ids.to(torch.int32).contiguous()
        w32 = topk_weights.to(torch.float32).contiguous()
        if slot is not None:
            # Copy by address. The shared buffers are registered with the driver, but the
            # tensor library keeps its own record of what is page-locked and a mapping it
            # did not allocate is not in it -- so an ordinary copy here is treated as a
            # pageable one, which is synchronous, which a capture rejects.
            stream = torch.cuda.current_stream().cuda_stream
            for buf, src in (
                (self._io["x"], hidden_states.contiguous()),
                (self._io["ids"], ids32),
                (self._io["w"], w32),
            ):
                self._memops.memcpy_async(
                    stream, buf.tensor.data_ptr(), src.data_ptr(),
                    src.numel() * src.element_size(),
                )
        else:
            self._io["x"].tensor[:bs].copy_(hidden_states)
            self._io["ids"].tensor[:bs].copy_(ids32)
            self._io["w"].tensor[:bs].copy_(w32)
        if slot is None:
            # Polled fallback: correct, and the reason capture stays off when it is in use.
            self._flags["slot_layer"].tensor[0] = layer_id
            self._flags["slot_bs"].tensor[0] = bs
            self._flags["done"].tensor[0] = 0
            self._flags["ready"].tensor[0] = 1
            return (0, bs, hidden_states.device, False)

        # Both halves go on the stream, so a capture records them and a replay drives the
        # worker for real instead of reading whatever the buffer held last.
        self._memops.memop_submit(
            torch.cuda.current_stream().cuda_stream,
            self._flags["done"].tensor.data_ptr(),
            self._flags["ready"].tensor.data_ptr(),
            slot,
        )
        return (slot, bs, hidden_states.device, True)

    def decode_sync(self, pending) -> torch.Tensor:
        """Wait for the work :meth:`decode_submit` rang for, and bring the result back."""
        slot, bs, device, on_stream = pending
        if not on_stream:
            self._await_slot(slot, _STEP_TIMEOUT_S)
            return self._io["y"].tensor[:bs].to(device)
        # A front-end wait: this stream's later nodes do not run until the worker reports
        # done, and the wait itself is a node like any other.
        stream = torch.cuda.current_stream().cuda_stream
        self._memops.memop_sync(stream, self._flags["done"].tensor.data_ptr(), slot)
        out = torch.empty(
            (bs, self._io["y"].tensor.shape[1]), dtype=self._io["y"].tensor.dtype,
            device=device,
        )
        self._memops.memcpy_async(
            stream, out.data_ptr(), self._io["y"].tensor.data_ptr(),
            out.numel() * out.element_size(),
        )
        return out

    def _await_slot(self, slot: int, timeout: float) -> None:
        """Poll one done flag, the fallback for a shape with no slot of its own."""
        done = self._flags["done"].tensor
        deadline = time.time() + timeout
        while not int(done[slot]):
            rc = self._proc.poll() if self._proc is not None else None
            if rc is not None:
                raise RuntimeError(
                    f"MoE worker for device {self.device_index} exited with {rc} during "
                    f"decode: {self._child_error()}"
                )
            if time.time() > deadline:
                raise TimeoutError(
                    f"MoE worker on device {self.device_index} did not answer within "
                    f"{timeout:.0f}s"
                )
            time.sleep(0)

    def serves(self, layer_id: int) -> bool:
        """Whether this worker can reach ``layer_id``'s weights at all."""
        return layer_id in self.layers

    def slot_stats(self) -> dict:
        """Hits and misses the worker's slot cache has seen, for the parent to report.

        Read from the control buffer rather than asked for over the doorbell: the worker
        writes them at the end of every step, so this costs a shared-memory read and never
        interrupts the worker to answer.
        """
        flags = self._ctl.tensor
        hits, misses = int(flags[_HITS]), int(flags[_MISSES])
        looks = hits + misses
        return {
            "slots": self.slots,
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / looks) if looks else 0.0,
        }

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._ctl.tensor[_STOP] = 1
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        for buf in self._bufs:
            buf.close()
        self._bufs = []
        if self._log is not None:
            self._log.close()
            self._log = None
        if not self._keep_spec:
            for path in (self._spec_path, self._log_path):
                try:
                    os.unlink(path)
                except (FileNotFoundError, OSError):
                    pass

    def __enter__(self) -> "WorkerMoeExecutor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["WorkerMoeExecutor"]
