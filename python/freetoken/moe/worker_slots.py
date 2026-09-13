"""An LRU slot cache over expert rows, for a worker process serving one MoE layer.

The main device has one of these already, built into :class:`~freetoken.moe.offload_cache.
OffloadMoeCache` as device-side kernels fused into the layer's forward. A worker cannot use
that one: it lives on the engine's device, in the engine's process, wired to the engine's
graph capture. So a worker that wanted experts on demand had no way to ask for them, and
took the only other option -- staging its whole layer onto its device at startup and never
moving anything again. That makes a worker layer a *static* assignment, outside the
economics every other layer is subject to.

This is the missing half. The worker keeps a fixed number of device slots and fills them
from the shared host bank as routing asks for experts, evicting least-recently-used ones,
exactly as the main device does. Two differences follow from where it runs:

* **Bookkeeping is on the host.** The worker already reads this step's expert ids out of
  shared memory on the CPU and already synchronises once per step, so deciding placement in
  Python costs a few microseconds of work that was going to be spent waiting anyway. The
  main device cannot do this -- its decision has to be inside a captured graph -- which is
  why its version is a device kernel and this one is not.
* **Slots are addressed by remapping ids, not by moving rows.** The grouped GEMV indexes
  whatever tensor it is handed by row, so a cache hit costs nothing at all: the expert's id
  is rewritten to its slot and the same kernel runs unchanged.

Sizing is the caller's business. ``slots`` equal to the expert count degenerates to holding
everything, which is the old behaviour minus the startup copy of experts routing never
asks for. Fewer slots trades hit rate for device memory, which is the whole point of having
the option.
"""

from __future__ import annotations

import torch

# A single token's experts are read by one launch and cannot be split, so top_k is the
# hard floor. Anything above that is a choice: a step whose demand exceeds the cache is
# served in several launches instead (see :meth:`WorkerSlotCache.partition`).
MIN_SLOTS_MESSAGE = (
    "worker slot cache too small: one token routes to {wanted} experts but the cache holds "
    "{slots}. A single token's experts are read by one launch and cannot be split across "
    "refills, so --moe-worker-slots can never be below top_k."
)


class WorkerSlotCache:
    """Device slots backed by host expert banks, filled on demand and evicted LRU.

    ``host_banks`` maps a bank name to a host tensor shaped ``[num_experts, ...]``; the
    device slot array for each mirrors it with ``slots`` rows instead. All banks share one
    placement decision -- an expert is present in all of them or none -- because routing
    names an expert, not a bank.
    """

    __slots__ = ("device", "num_experts", "slots", "host", "dev",
                 "_slot_of", "_expert_of", "_used", "_clock", "hits", "misses", "fills")

    def __init__(
        self,
        host_banks: dict[str, torch.Tensor],
        *,
        slots: int,
        device: torch.device,
    ) -> None:
        assert host_banks, "a slot cache needs at least one bank"
        counts = {t.shape[0] for t in host_banks.values()}
        assert len(counts) == 1, f"banks disagree on expert count: {counts}"
        self.num_experts = counts.pop()
        self.slots = max(1, min(int(slots), self.num_experts))
        self.device = device
        self.host = host_banks
        self.dev = {
            name: torch.empty((self.slots, *t.shape[1:]), dtype=t.dtype, device=device)
            for name, t in host_banks.items()
        }
        self._slot_of = [-1] * self.num_experts  # expert id -> slot, -1 = absent
        self._expert_of = [-1] * self.slots      # slot -> expert id, -1 = free
        self._used = [0] * self.slots            # slot -> clock value at last touch
        self._clock = 0
        self.hits = 0
        self.misses = 0
        self.fills = 0

    @property
    def holds_everything(self) -> bool:
        """True when eviction can never happen, so the cache is a residency plan."""
        return self.slots >= self.num_experts

    def _victim(self, keep: set[int]) -> int:
        """Least-recently-used slot that this step does not still need.

        ``keep`` holds the slots already claimed for the step in progress. Evicting one of
        those would free a row the very same kernel launch is about to read, which is why
        the caller's demand -- not the cache size alone -- sets the floor on slots.
        """
        victim, oldest = -1, None
        for slot in range(self.slots):
            if slot in keep:
                continue
            if self._expert_of[slot] < 0:
                return slot  # a free slot is always the best victim
            if oldest is None or self._used[slot] < oldest:
                victim, oldest = slot, self._used[slot]
        return victim

    def partition(self, expert_ids: torch.Tensor) -> list[tuple[int, int]]:
        """Split ``[tokens, top_k]`` ids into token ranges the cache can hold one at a time.

        Sizing the cache would be pointless if a step's demand set the floor: prefill hands
        this device hundreds of tokens at once, whose union is the whole expert set, so any
        cache smaller than the layer would be rejected and the option would collapse back
        to holding everything. Tokens are independent, though -- each is a separate row of
        the grouped GEMV -- so a step too wide for the cache becomes several launches over
        contiguous ranges, each within its reach.

        Ranges are grown greedily, which keeps the common case (everything fits) to exactly
        one range and no copying at all.
        """
        rows = expert_ids.tolist()
        if not rows:
            return []
        ranges: list[tuple[int, int]] = []
        start, union = 0, set()
        for token, row in enumerate(rows):
            experts = {int(e) for e in row if int(e) >= 0}
            if len(experts) > self.slots:
                raise RuntimeError(
                    MIN_SLOTS_MESSAGE.format(wanted=len(experts), slots=self.slots)
                )
            if token > start and len(union | experts) > self.slots:
                ranges.append((start, token))
                start, union = token, set(experts)
            else:
                union |= experts
        ranges.append((start, len(rows)))
        return ranges

    def ensure(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Make every expert in ``expert_ids`` resident; return the ids remapped to slots.

        ``expert_ids`` is a host tensor of shape ``[tokens, top_k]``. The returned tensor
        has the same shape on ``device`` and indexes the slot arrays, so it can be passed
        straight to the grouped GEMV in place of the original ids.
        """
        rows = expert_ids.tolist()
        # A negative id marks a route another executor owns. Its weight is zero, so what it
        # computes cannot matter -- but it still has to name a slot the kernel can read, and
        # Python would happily let -1 index the last one. Point them at slot 0 instead of
        # letting a sentinel wander into the cache as if it were an expert.
        wanted = {int(e) for row in rows for e in row if int(e) >= 0}
        if len(wanted) > self.slots:
            raise RuntimeError(MIN_SLOTS_MESSAGE.format(wanted=len(wanted), slots=self.slots))

        keep: set[int] = set()
        missing = []
        for expert in sorted(wanted):
            slot = self._slot_of[expert]
            if slot >= 0:
                self.hits += 1
                keep.add(slot)
            else:
                self.misses += 1
                missing.append(expert)

        for expert in missing:
            slot = self._victim(keep)
            evicted = self._expert_of[slot]
            if evicted >= 0:
                self._slot_of[evicted] = -1
            for name, dev_bank in self.dev.items():
                dev_bank[slot].copy_(self.host[name][expert], non_blocking=False)
            self._expert_of[slot] = expert
            self._slot_of[expert] = slot
            self.fills += 1
            keep.add(slot)

        # Touch after filling so a freshly filled slot is the most recent, not the oldest.
        for expert in wanted:
            self._clock += 1
            self._used[self._slot_of[expert]] = self._clock

        remapped = [
            [self._slot_of[int(e)] if int(e) >= 0 else 0 for e in row] for row in rows
        ]
        return torch.tensor(remapped, dtype=expert_ids.dtype, device=self.device)

    def stats(self) -> tuple[int, int]:
        """``(hits, misses)`` since construction, for the parent to report."""
        return self.hits, self.misses


class WorkerInPlaceBanks:
    """The layer's experts read where they already are, with nothing copied anywhere.

    A slot cache exists to make a small fast memory serve a large working set. Where the
    device can address the host bank directly, there is no smaller memory and no working
    set to manage: every expert is reachable at all times, so this is a cache with no
    capacity limit and no misses -- which is why it answers the same questions and always
    says yes.

    Measured against copying into the device's own memory first: on an integrated device,
    whose memory is the same DRAM, reading in place costs 16% and copying costs a full
    transfer plus a second residency, so in place wins outright. On a discrete device the
    trade reverses as soon as an expert is reused, which is what its slot cache is for.
    Nothing here decides that -- the caller picks, and this is the half that does not copy.

    The bank tensors name memory the allocator never handed out, so the mappings they point
    at must outlive them; ``host_banks`` is held for exactly that reason.
    """

    __slots__ = ("device", "num_experts", "slots", "host", "dev", "hits", "misses", "fills",
                 "_registered")

    def __init__(self, host_banks: dict[str, torch.Tensor], *, device: torch.device) -> None:
        from freetoken.kernel.pinned import (
            device_ptr,
            host_register,
            tensor_from_device_ptr,
        )

        assert host_banks, "an in-place reader needs at least one bank"
        counts = {t.shape[0] for t in host_banks.values()}
        assert len(counts) == 1, f"banks disagree on expert count: {counts}"
        self.num_experts = counts.pop()
        self.device = device
        self.host = host_banks  # keeps the mappings the device tensors point into alive
        self.dev = {}
        # Addresses this instance registered, so release() can hand them back. The device
        # can hold only so many host registrations at once (an iGPU runs out around a dozen
        # layers), and the worker is offered every layer -- so they are a resource with a
        # capacity, and a capacity has to be returnable.
        self._registered: list[int] = []
        registered_bytes = 0
        registered = 0
        for name, tensor in host_banks.items():
            nbytes = tensor.numel() * tensor.element_size()
            try:
                host_register(tensor.data_ptr(), nbytes)
            except Exception as exc:
                # Say which bank, how far in, and what it looked like. "invalid argument"
                # covers several unrelated causes and the numbers are what separate them;
                # without them the same message appears for an unaligned address, a size
                # the runtime will not take, and a limit reached several banks earlier.
                self.release()
                raise RuntimeError(
                    f"{exc}\n  bank {name!r} shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} {nbytes} bytes "
                    f"(page offset {tensor.data_ptr() % 4096}, "
                    f"{nbytes / 4096:.2f} pages); "
                    f"{registered} banks / {registered_bytes / 2**20:.0f} MiB already "
                    f"registered on this device by this process"
                ) from None
            self._registered.append(tensor.data_ptr())
            registered += 1
            registered_bytes += nbytes
            self.dev[name] = tensor_from_device_ptr(
                device_ptr(tensor), tensor.shape, tensor.dtype, device.index or 0
            )
        self.slots = self.num_experts
        self.hits = 0
        self.misses = 0
        self.fills = 0

    def release(self) -> None:
        """Give the host registrations back, leaving the shared mapping itself alone.

        Only the registration is scarce; the mapping is address space and the engine's bank
        is not ours to unmap. After this the instance is spent -- the caller drops it and
        builds a new one when that layer comes back, which costs a registration and no copy.
        """
        from freetoken.kernel.pinned import host_unregister

        for addr in self._registered:
            try:
                host_unregister(addr)
            except Exception:
                pass  # already gone, or the runtime is tearing down: nothing left to free
        self._registered.clear()
        self.dev.clear()

    @property
    def holds_everything(self) -> bool:
        return True

    def partition(self, expert_ids: torch.Tensor) -> list[tuple[int, int]]:
        """One range, always: there is no capacity to run out of."""
        return [(0, expert_ids.shape[0])] if expert_ids.shape[0] else []

    def ensure(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Nothing to fetch. An expert's id is already its row, so the ids pass through.

        Routes another executor owns still have to name a readable row -- their weight is
        zero, so which row cannot matter, but a negative index would wander off the front
        of the bank.
        """
        self.hits += int((expert_ids >= 0).sum())
        return expert_ids.clamp_min(0).to(self.device)

    def stats(self) -> tuple[int, int]:
        return self.hits, self.misses
