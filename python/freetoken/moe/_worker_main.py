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
import os
import sys
from collections import OrderedDict
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
    "float8_e4m3fn": torch.float8_e4m3fn,
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
        self._in_place_warned = False
        self._in_place_noted = False
        self._maps: dict[int, dict] = {}
        # Ordered by last use: a slot cache costs device memory per layer, and a worker
        # offered every layer would otherwise hold one for each. The oldest are dropped to
        # make room (see cache_for), so the footprint follows this device's memory rather
        # than the number of layers the engine may route here.
        self._caches: "OrderedDict[int, WorkerSlotCache]" = OrderedDict()
        self._evicted = [0, 0]  # hits, misses of caches already dropped
        any_name = next(iter(spec))
        self.layers = sorted(int(k) for k in spec[any_name]["paths"])

    def cache_for(self, layer_id: int, device) -> WorkerSlotCache:
        cache = self._caches.get(layer_id)
        if cache is not None:
            self._caches.move_to_end(layer_id)
            return cache
        tensors = {}
        for name, entry in self._spec.items():
            path = entry["paths"][str(layer_id)]
            tensors[name] = open_shared(
                path, tuple(entry["shape"]), _DTYPES[entry["dtype"]]
            ).tensor
        self._maps[layer_id] = tensors
        cache = None
        if self._in_place:
            # Nothing is copied and nothing is evicted: the device reads the engine's own
            # bank where it lies. Costs no device memory, so every layer can have one.
            cache = self._new_in_place(tensors, device, layer_id)
            if cache is not None and self._in_place_noted is False:
                print("worker: reading banks in place (GTT, no device copy)",
                      file=sys.stderr, flush=True)
                self._in_place_noted = True
        if cache is None:
            cache = self._new_slot_cache(tensors, device, layer_id)
        self._caches[layer_id] = cache
        return cache

    def _new_in_place(self, tensors: dict, device, layer_id: int):
        """An in-place reader for one layer, giving back other layers' host registrations to fit.

        Registering host pages for a device is itself a bounded resource -- on this iGPU it
        runs out around the thirteenth layer, well before any memory does, because nothing
        here allocates memory at all. Treating that limit as "this device cannot read in
        place" was the wrong reading: it is "not all of them at once". So the layer used
        longest ago hands its registrations back and the new one takes them, exactly as the
        slot path recycles memory. Nothing is copied either way; a layer that returns
        re-registers pages it never stopped sharing.

        Only a failure with no registration left to reclaim means the device truly will not
        do this, and then the caller falls back to copying.
        """
        while True:
            try:
                return WorkerInPlaceBanks(tensors, device=device)
            except Exception as exc:
                victim = next(
                    (lid for lid, c in self._caches.items() if isinstance(c, WorkerInPlaceBanks)),
                    None,
                )
                if victim is None:
                    if self._in_place_warned is False:
                        print(f"worker: cannot read banks in place on layer {layer_id} "
                              f"({type(exc).__name__}: {exc}); copying instead",
                              file=sys.stderr, flush=True)
                        self._in_place_warned = True
                    self._in_place = False
                    return None
                old = self._caches.pop(victim)
                self._evicted[0] += old.hits
                self._evicted[1] += old.misses
                old.release()
                print(f"worker: released layer {victim}'s host registration to map layer "
                      f"{layer_id} ({len(self._caches)} layers mapped)",
                      file=sys.stderr, flush=True)

    def _new_slot_cache(self, tensors: dict, device, layer_id: int) -> WorkerSlotCache:
        """A slot cache for one layer, dropping the least recently used ones to fit.

        --moe-worker-slots is a per-layer count, and the engine may offer this worker every
        offloaded layer; holding a cache for each is that count times the layer count, which
        on an iGPU sharing 16 GiB with the host runs out around layer thirty. What the device
        can hold is not a number this side can compute -- the banks, the engine's own
        allocations and the runtime all move -- so it is discovered: allocate, and on OOM give
        back the layer used longest ago and try again. A layer that comes back rebuilds its
        cache from the shared bank, which is a refill, not a reload.
        """
        slots = self._slots or tensors[next(iter(tensors))].shape[0]
        while True:
            try:
                return WorkerSlotCache(tensors, slots=slots, device=device)
            except torch.OutOfMemoryError:
                if not self._caches:
                    raise
                old_id, old = self._caches.popitem(last=False)
                self._evicted[0] += old.hits
                self._evicted[1] += old.misses
                del old
                torch.cuda.empty_cache()
                print(f"worker: dropped layer {old_id}'s slot cache to fit layer "
                      f"{layer_id} ({len(self._caches)} layers resident)",
                      file=sys.stderr, flush=True)

    @property
    def in_place_ok(self) -> bool:
        """Whether the last cache_for stayed on the no-copy path."""
        return self._in_place

    def stats(self) -> tuple[int, int]:
        hits = self._evicted[0] + sum(c.hits for c in self._caches.values())
        misses = self._evicted[1] + sum(c.misses for c in self._caches.values())
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

    expert_call = _expert_call(_SPEC.get("quant_format", "q4_0"), _SPEC.get("ggml_type"))
    activation = _SPEC["activation"]
    act_fn = _resolve_activation(
        activation,
        os.environ.get("FREETOKEN_WORKER_ACTIVATION")
        or _SPEC.get("activation_backend", "auto"),
    )
    # Announce readiness only once the kernels are actually loaded: the first launch JIT
    # compiles, and a parent that started timing before that would blame the first token.
    _warm(library.cache_for(library.layers[0], device), io, device, activation,
          expert_call, act_fn)

    # Map what fits, and say so. Reading in place needs the runtime to register the bank's
    # pages for this device, and that budget is bounded and one-way -- freeing a
    # registration does not give it back. Finding the line here, once, is what keeps a step
    # from discovering it: a layer the engine is told about is one this worker can answer
    # for, and the ones past the line are divided between the main device and the CPU by
    # the same placement that divides everything else. Nothing is pinned by hand either
    # way; the device's own limit picks how many, and demand picks which.
    servable = flags["servable"].tensor
    servable.zero_()
    mapped = 0
    for layer_id in library.layers:
        try:
            library.cache_for(layer_id, device)
        except Exception as exc:
            print(f"worker: stopping at layer {layer_id}: {type(exc).__name__}: "
                  f"{str(exc)[:120]}", file=sys.stderr, flush=True)
            break
        if not library.in_place_ok:
            break  # fell out of in place: past this point it would copy, which is not ours
        servable[layer_id] = 1
        mapped += 1
    print(f"worker: can serve {mapped} of {len(library.layers)} layers in place",
          file=sys.stderr, flush=True)
    ctl.tensor[_UP] = 1  # the parent waits for this before its first submit

    _TRACE = os.environ.get("FREETOKEN_WORKER_TRACE") == "1"
    _traced = 0
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
        if _TRACE and _traced < 400:
            print(f"worker: step {_traced} layer {int(slot_layer[slot])} "
                  f"|x|={float(x.float().abs().sum()):.4f} "
                  f"|w|={float(w.float().abs().sum()):.4f} "
                  f"ids={io['ids'].tensor[:bs].reshape(-1)[:6].tolist()}",
                  file=sys.stderr, flush=True)
            _traced += 1
        # The ids stay on the host for one more moment: placement is decided here, and what
        # the kernel receives is slot numbers, not expert ids. A step wider than the cache
        # becomes several launches -- one range at a time, each refilled before it runs, so
        # the cache size is a memory choice and never a limit on batch width.
        host_ids = io["ids"].tensor[:bs]
        for lo, hi in cache.partition(host_ids):
            ids = cache.ensure(host_ids[lo:hi])
            out = expert_call(x[lo:hi], cache.dev, w[lo:hi], ids, activation, act_fn)
            io["y"].tensor[lo:hi].copy_(out)  # cross-device copy; syncs on this stream
            if _TRACE and _traced <= 400:
                print(f"worker:   -> |y|={float(out.float().abs().sum()):.4f} "
                      f"rows={lo}:{hi}", file=sys.stderr, flush=True)
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
    if torch_fn is None:
        return None
    # A name this table does not carry is still served by a compiled activation inside
    # fused_experts_gguf, so "not listed here" is not "not compiled" -- returning None on
    # that basis is how an uncompilable device used to reach the kernel anyway. Probe with
    # whatever compiled activation is reachable instead: the failure being guarded against
    # ("unsupported target") is a property of the device, not of one kernel.
    probe_fn = kernel_fn or next(iter(_ACT.values()), None)
    if probe_fn is None:
        return None
    probe = torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda")
    try:
        probe_fn(probe)
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"worker: compiled {name} activation unavailable on this device "
            f"({type(exc).__name__}: {str(exc).splitlines()[-1][:160]}); using torch",
            file=sys.stderr, flush=True,
        )
        return torch_fn
    return None


def _expert_call(quant_format: str, ggml_type):
    """Bind this checkpoint's expert kernel to the worker's one calling shape.

    Returns ``f(x, banks, w, ids, activation, act_fn)``. The bank names a format needs are
    known only to the format, so they are named here and nowhere else in the worker: the
    decode loop and the warm-up both stay format-agnostic, and adding a format is adding a
    branch here rather than threading a second set of arguments through both.
    """
    if quant_format == "nvfp4":
        from freetoken.moe.fused_nvfp4_vec import fused_experts_nvfp4_vec

        # The kernel's layout names its banks by role (gate_up, gate_up_scale, ...); FTW
        # files and the older per-format schema call the packed ones gate_up_packed and
        # down_packed. Take whichever this checkpoint's banks were published under.
        roles = (("gate_up", "gate_up_packed"), ("gate_up_scale", "gate_up_scale"),
                 ("gate_up_global", "gate_up_global"), ("down", "down_packed"),
                 ("down_scale", "down_scale"), ("down_global", "down_global"))

        def call(x, banks, w, ids, activation, act_fn):
            return fused_experts_nvfp4_vec(
                x, *(banks[r] if r in banks else banks[legacy] for r, legacy in roles),
                w, ids, activation, act_fn,
            )

        return call

    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    qt = int(ggml_type)

    def call(x, banks, w, ids, activation, act_fn):
        return fused_experts_gguf(
            x, banks["gate_up"], banks["down"], w, ids, activation, qt, act_fn
        )

    return call


def _warm(cache, io, device, activation, fn, act_fn=None) -> None:
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
        fn(x, cache.dev, w, ids, activation, act_fn)
        torch.cuda.synchronize(device)
    except Exception as exc:  # a warm-up failure is the parent's problem, not a crash here
        print(f"worker warmup failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
