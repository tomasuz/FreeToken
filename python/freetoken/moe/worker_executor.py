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

_READY, _DONE, _STOP = 0, 1, 2
_CTL_SLOTS = 3

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
    ) -> None:
        self.device_index = device_index
        self._proc: subprocess.Popen | None = None
        self._bufs: list = []
        self._keep_spec = False
        self._log = None
        self._log_path = ""

        gate_up = banks["gate_up"]
        down = banks["down"]
        h = hidden_size if hidden_size is not None else down.shape[1]
        k = top_k if top_k is not None else 1
        tag = uuid.uuid4().hex[:12]

        def share(name: str, src: torch.Tensor | None, shape, dtype):
            buf = create_shared(f"freetoken-{tag}-{name}", tuple(shape), dtype)
            if src is not None:
                buf.tensor.copy_(src)
            self._bufs.append(buf)
            return buf

        self._ctl = share("ctl", None, (_CTL_SLOTS,), torch.int64)
        self._ctl.tensor.zero_()
        bank_bufs = {
            "gate_up": share("gate_up", gate_up, gate_up.shape, gate_up.dtype),
            "down": share("down", down, down.shape, down.dtype),
        }
        self._io = {
            "x": share("x", None, (max_batch, h), torch.bfloat16),
            "ids": share("ids", None, (max_batch, k), torch.int32),
            "w": share("w", None, (max_batch, k), torch.float32),
            "y": share("y", None, (max_batch, h), torch.bfloat16),
        }
        # Not registered while the handoff is a plain copy: the pages are shared with a
        # process driving a different device, and registering them here would be an
        # optimisation whose only beneficiary is a stream-async copy this cut does not do.
        # The stream-memop handshake will need it and can add it deliberately.

        spec = {
            "control": self._entry(self._ctl),
            "banks": {n: self._entry(b) for n, b in bank_bufs.items()},
            "io": {n: self._entry(b) for n, b in self._io.items()},
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
            self._await_flag(_DONE, -1, _START_TIMEOUT_S, "start")
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
        self._io["x"].tensor[:bs].copy_(hidden_states)
        self._io["ids"].tensor[:bs].copy_(topk_ids.to(torch.int32))
        self._io["w"].tensor[:bs].copy_(topk_weights.to(torch.float32))

        flags = self._ctl.tensor
        flags[_DONE] = 0
        flags[_READY] = bs  # doorbell
        self._await_flag(_DONE, bs, _STEP_TIMEOUT_S, "decode")
        return self._io["y"].tensor[:bs].to(hidden_states.device)

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
