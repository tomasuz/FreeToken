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

    # Pinned so the one-off staging copy below runs at full rate. Registration also gives
    # this process a device address for the mapping, which is what a later zero-copy read
    # would need -- see the note on materialisation below.
    for b in banks.values():
        b.pin()

    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    ggml_type = int(_SPEC["ggml_type"])
    activation = _SPEC["activation"]
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
    _warm(gate_up, down, io, device, activation, ggml_type, fused_experts_gguf)
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

        out = fused_experts_gguf(x, gate_up, down, w, ids, activation, ggml_type)
        io["y"].tensor[:bs].copy_(out)  # cross-device copy; syncs on this stream
        torch.cuda.synchronize(device)

        flags[_READY] = 0
        flags[_DONE] = bs


def _warm(gate_up, down, io, device, activation, ggml_type, fn) -> None:
    """One throwaway launch so the JIT compile lands before the parent starts timing."""
    x = io["x"].tensor[:1].to(device)
    ids = io["ids"].tensor[:1].to(device)
    w = io["w"].tensor[:1].to(device)
    try:
        fn(x, gate_up, down, w, ids, activation, ggml_type)
        torch.cuda.synchronize(device)
    except Exception as exc:  # a warm-up failure is the parent's problem, not a crash here
        print(f"worker warmup failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
