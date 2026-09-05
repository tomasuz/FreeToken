"""Entry point for a MoE worker process bound to one accelerator.

Runs as ``python -m freetoken.moe._worker_main <spec.json>``. Everything it needs -- which
device to bind, which shared buffers to map, which expert format to serve -- arrives in the
spec, so the module holds no knowledge of the machine it lands on.

The loop is a doorbell, not a queue: the parent writes this step's activations into the
shared input buffers, bumps ``ready``, and waits on ``done``. Polling (rather than a pipe
read) is what keeps the handoff at memory latency; the parent side can then drive it with
stream memory operations and never leave the GPU's front end.

Why a separate process at all: the environment that selects an accelerator's architecture
or runtime libraries is read once per process, so a machine whose devices disagree about it
cannot serve them all from one. The parent applies this worker's overrides to the child's
environment before spawn; by the time this module imports torch, they are simply the truth.
"""

from __future__ import annotations

import json
import sys
import time

# Set before torch is imported: the accelerator runtime latches its view of the environment
# on first use, so anything applied afterwards is silently too late.
_SPEC = json.loads(open(sys.argv[1]).read())

import torch  # noqa: E402

from freetoken.moe.shared_host import open_shared  # noqa: E402

# Flag layout in the control buffer, one int64 each. Kept adjacent so the parent can hand
# both addresses to a single stream memop pair.
_READY, _DONE, _STOP = 0, 1, 2

_DTYPES = {
    "uint8": torch.uint8,
    "int32": torch.int32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "int64": torch.int64,
}


def _map(entry: dict):
    return open_shared(entry["path"], tuple(entry["shape"]), _DTYPES[entry["dtype"]])


def main() -> int:
    device = torch.device("cuda", 0)  # the spec restricted visibility, so ours is index 0
    torch.cuda.set_device(device)

    ctl = _map(_SPEC["control"])
    banks = {name: _map(e) for name, e in _SPEC["banks"].items()}
    io = {name: _map(e) for name, e in _SPEC["io"].items()}

    # The bank mappings are deliberately NOT registered with the runtime. Registering a
    # shared mapping that another process also holds is not something the runtime promises
    # anything about, and it bought only a faster one-off staging copy at startup. The
    # zero-copy read this would be a prerequisite for is a separate piece of work, and can
    # bring its own registration when it is written.

    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    ggml_type = int(_SPEC["ggml_type"])
    activation = _SPEC["activation"]
    act_fn = _resolve_activation(activation, _SPEC.get("activation_backend", "auto"))
    flags = ctl.tensor

    # First cut: stage the layer's banks onto the device once, at startup. The grouped
    # GEMV takes torch tensors, and a registered host mapping has a device address but no
    # tensor that names it, so reading the shared pages in place needs a kernel entry that
    # accepts a raw pointer. Until then this costs one copy of the layer on the worker's
    # device -- which for a device whose memory *is* host memory means the bytes are
    # resident twice. Correctness first; the zero-copy read is the obvious follow-up.
    gate_up = banks["gate_up"].tensor.to(device)
    down = banks["down"].tensor.to(device)
    # Announce readiness only once the kernels are actually loaded: the first launch JIT
    # compiles, and a parent that started timing before that would blame the first token.
    _warm(gate_up, down, io, device, activation, ggml_type, fused_experts_gguf, act_fn)
    flags[_DONE] = -1  # "worker is up"; the parent waits for this before its first submit

    while True:
        if flags[_STOP]:
            return 0
        if not flags[_READY]:
            time.sleep(0)  # yield without leaving the run queue
            continue
        bs = int(flags[_READY])  # the batch size doubles as the doorbell; 0 means idle
        x = io["x"].tensor[:bs].to(device, non_blocking=False)
        ids = io["ids"].tensor[:bs].to(device, non_blocking=False)
        w = io["w"].tensor[:bs].to(device, non_blocking=False)

        out = fused_experts_gguf(x, gate_up, down, w, ids, activation, ggml_type, act_fn)
        io["y"].tensor[:bs].copy_(out)  # cross-device copy; syncs on this stream
        torch.cuda.synchronize(device)

        flags[_READY] = 0
        flags[_DONE] = bs


def _resolve_activation(name: str, backend: str):
    """Pick the activation implementation this device can actually run.

    ``"kernel"`` and ``"torch"`` say so outright. ``"auto"`` (the default) tries the
    compiled kernel on a token-sized input and falls back to torch if it does not compile,
    which is how a device Triton has no backend for stays usable: losing a fused kernel
    costs speed, and refusing the device costs the device.

    The probe runs here, once, rather than at the first real step -- a compile failure in
    the middle of a decode would surface as a stalled worker instead of a clear line at
    startup. Only a compile-time failure is caught: a launch failure means the device is
    broken in a way a different activation will not repair, so it propagates.
    """
    from freetoken.layers import activation_torch

    torch_fn = activation_torch.BY_NAME.get(name)
    if backend == "torch":
        if torch_fn is None:
            raise ValueError(f"no torch activation for {name!r}")
        return torch_fn
    if backend == "kernel":
        return None  # fused_experts_gguf keeps its own lookup
    if backend != "auto":
        raise ValueError(f"unknown activation_backend {backend!r}")

    from freetoken.moe.fused_q4_0 import _ACT

    kernel_fn = _ACT.get(name)
    if kernel_fn is None or torch_fn is None:
        return None
    probe = torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda")
    try:
        kernel_fn(probe)
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"worker: compiled {name} activation unavailable on this device "
            f"({type(exc).__name__}: {str(exc).splitlines()[-1][:160]}); using torch",
            file=sys.stderr, flush=True,
        )
        return torch_fn
    return None


def _warm(gate_up, down, io, device, activation, ggml_type, fn, act_fn=None) -> None:
    """One throwaway launch so the JIT compile lands before the parent starts timing."""
    # Built here rather than sliced from the shared buffers: a warm-up wants representative
    # shapes, not whatever those pages happen to hold, and routing ids that are merely
    # "whatever was in memory" are not a launch worth compiling against.
    x = torch.zeros(1, io["x"].tensor.shape[1], dtype=io["x"].tensor.dtype, device=device)
    ids = torch.zeros(1, io["ids"].tensor.shape[1], dtype=torch.int32, device=device)
    w = torch.ones(1, io["w"].tensor.shape[1], dtype=torch.float32, device=device)
    try:
        fn(x, gate_up, down, w, ids, activation, ggml_type, act_fn)
        torch.cuda.synchronize(device)
    except Exception as exc:  # a warm-up failure is the parent's problem, not a crash here
        print(f"worker warmup failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
