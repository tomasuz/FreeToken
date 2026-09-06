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
from freetoken.moe.worker_slots import (  # noqa: E402
    WorkerInPlaceBanks,
    WorkerSlotCache,
)

# Flag layout in the control buffer, one int64 each. Kept adjacent so the parent can hand
# both addresses to a single stream memop pair.
_STOP, _HITS, _MISSES, _UP = 0, 1, 2, 3

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



class _BankLibrary:
    """Every layer this worker can serve, mapped lazily, with a slot cache each.

    Mapping all of them up front would be wasteful only in address space, but opening
    files this process may never read is still work, so a layer arrives the first time it
    is asked for. Its cache arrives with it: a layer nothing routes to costs nothing on
    this device, which is what lets the engine offer this worker every layer and decide per
    step rather than at start-up.
    """

    def __init__(self, spec: dict, slots: int, in_place: bool) -> None:
        self._spec = spec
        self._slots = slots
        self._in_place = in_place
        self._maps: dict[int, dict] = {}
        self._caches: dict[int, WorkerSlotCache] = {}
        any_name = next(iter(spec))
        self.layers = sorted(int(k) for k in spec[any_name]["paths"])

    def cache_for(self, layer_id: int, device) -> WorkerSlotCache:
        cache = self._caches.get(layer_id)
        if cache is not None:
            return cache
        tensors = {}
        for name, entry in self._spec.items():
            path = entry["paths"][str(layer_id)]
            tensors[name] = open_shared(
                path, tuple(entry["shape"]), _DTYPES[entry["dtype"]]
            ).tensor
        self._maps[layer_id] = tensors
        if self._in_place:
            # Nothing is copied and nothing is evicted: the device reads the engine's own
            # bank where it lies. Costs no device memory, so every layer can have one.
            cache = WorkerInPlaceBanks(tensors, device=device)
        else:
            cache = WorkerSlotCache(
                tensors, slots=self._slots or tensors[next(iter(tensors))].shape[0],
                device=device,
            )
        self._caches[layer_id] = cache
        return cache

    def stats(self) -> tuple[int, int]:
        hits = sum(c.hits for c in self._caches.values())
        misses = sum(c.misses for c in self._caches.values())
        return hits, misses


def main() -> int:
    device = torch.device("cuda", 0)  # the spec restricted visibility, so ours is index 0
    torch.cuda.set_device(device)

    ctl = _map(_SPEC["control"])
    io = {name: _map(e) for name, e in _SPEC["io"].items()}
    flags = {name: _map(e) for name, e in _SPEC["flags"].items()}
    # Every offloaded layer, mapped rather than copied. Mapping is address space until a
    # page is touched, so holding all of them costs nothing for the layers this worker is
    # never asked about -- and the ones it is asked about read the same pages the engine
    # holds rather than a second copy of them.
    library = _BankLibrary(
        _SPEC["banks"], int(_SPEC.get("slots") or 0), bool(_SPEC.get("read_in_place"))
    )

    # The bank mappings are deliberately NOT registered with the runtime. Registering a
    # shared mapping that another process also holds is not something the runtime promises
    # anything about, and it bought only a faster one-off staging copy at startup. The
    # zero-copy read this would be a prerequisite for is a separate piece of work, and can
    # bring its own registration when it is written.

    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    ggml_type = int(_SPEC["ggml_type"])
    activation = _SPEC["activation"]
    act_fn = _resolve_activation(activation, _SPEC.get("activation_backend", "auto"))
    # Announce readiness only once the kernels are actually loaded: the first launch JIT
    # compiles, and a parent that started timing before that would blame the first token.
    _warm(library.cache_for(library.layers[0], device), io, device, activation, ggml_type,
          fused_experts_gguf, act_fn)
    ctl.tensor[_UP] = 1  # the parent waits for this before its first submit

    ready, done = flags["ready"].tensor, flags["done"].tensor
    slot_layer, slot_bs = flags["slot_layer"].tensor, flags["slot_bs"].tensor
    control = ctl.tensor
    while True:
        if control[_STOP]:
            return 0
        # Scan for a raised slot rather than one doorbell word. The slot is the request:
        # its address encodes which layer and which batch size, which is what lets the
        # engine raise it from a captured graph -- a replay writes a fixed address and
        # cannot write a layer id into a shared word first.
        raised = (ready != 0).nonzero()
        if raised.numel() == 0:
            time.sleep(0)  # yield without leaving the run queue
            continue
        slot = int(raised[0])
        bs = int(slot_bs[slot])
        ready[slot] = 0
        cache = library.cache_for(int(slot_layer[slot]), device)
        x = io["x"].tensor[:bs].to(device, non_blocking=False)
        w = io["w"].tensor[:bs].to(device, non_blocking=False)
        # The ids stay on the host for one more moment: placement is decided here, and what
        # the kernel receives is slot numbers, not expert ids. A step wider than the cache
        # becomes several launches -- one range at a time, each refilled before it runs, so
        # the cache size is a memory choice and never a limit on batch width.
        host_ids = io["ids"].tensor[:bs]
        for lo, hi in cache.partition(host_ids):
            ids = cache.ensure(host_ids[lo:hi])
            out = fused_experts_gguf(
                x[lo:hi], cache.dev["gate_up"], cache.dev["down"], w[lo:hi],
                ids, activation, ggml_type, act_fn,
            )
            io["y"].tensor[lo:hi].copy_(out)  # cross-device copy; syncs on this stream
        torch.cuda.synchronize(device)

        control[_HITS], control[_MISSES] = library.stats()
        done[slot] = 1  # the engine's stream is waiting on exactly this word


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


def _warm(cache, io, device, activation, ggml_type, fn, act_fn=None) -> None:
    """One throwaway launch so the JIT compile lands before the parent starts timing."""
    # Built here rather than sliced from the shared buffers: a warm-up wants representative
    # shapes, not whatever those pages happen to hold, and routing ids that are merely
    # "whatever was in memory" are not a launch worth compiling against.
    x = torch.zeros(1, io["x"].tensor.shape[1], dtype=io["x"].tensor.dtype, device=device)
    w = torch.ones(1, io["w"].tensor.shape[1], dtype=torch.float32, device=device)
    # Route the warm-up through the cache too, so the fill path is exercised before the
    # first real step and its own failures surface here rather than mid-request.
    ids = cache.ensure(torch.zeros(1, io["ids"].tensor.shape[1], dtype=torch.int32))
    try:
        fn(x, cache.dev["gate_up"], cache.dev["down"], w, ids, activation, ggml_type, act_fn)
        torch.cuda.synchronize(device)
    except Exception as exc:  # a warm-up failure is the parent's problem, not a crash here
        print(f"worker warmup failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
