"""The CPU pool reports its own compute time to the placement.

Timed from the engine, the pool is charged the GPU's fetch and GEMM as well, reads back an
order of magnitude slow and is starved of work -- on GLM-4.5-Air that halved decode. What
has to hold is that the pool's own counters, and only they, reach the rate tracker: as
deltas, never re-counted, and not at all from an extension built before the counters.
"""

from __future__ import annotations

import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor
from freetoken.moe.placement import RateTracker


class _Ext:
    """The counters of the compiled pool: (tasks, routes, nanoseconds), cumulative."""

    def __init__(self):
        self.tasks = self.routes = self.ns = 0

    def run(self, routes, seconds):
        self.tasks += 1
        self.routes += routes
        self.ns += int(seconds * 1e9)

    def timing_counters(self):
        return self.tasks, self.routes, self.ns


def _executor(ext):
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)  # the pool itself needs the extension
    executor._ext = ext
    executor.self_timed = hasattr(ext, "timing_counters")
    executor._timing_last = (0, 0)
    return executor


def test_samples_are_deltas_and_are_not_counted_twice():
    ext = _Ext()
    executor = _executor(ext)

    ext.run(8, 0.004)
    ext.run(4, 0.002)
    assert executor.take_samples() == [(12, 0.006)]
    assert executor.take_samples() == []  # nothing new since the last ask

    ext.run(3, 0.0015)
    assert executor.take_samples() == [(3, 0.0015)]


def test_an_extension_without_counters_keeps_the_old_measurement():
    executor = _executor(object())

    assert executor.self_timed is False
    assert executor.take_samples() == []


def test_the_placement_reads_the_cpu_from_its_own_samples():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=8, cache_size=8,
        device=torch.device("cpu"), quant_format="q4_0", prefill_overlap=False,
    )
    cache.rate_tracker = RateTracker()
    ext = _Ext()
    cache.cpu_executor = _executor(ext)

    ext.run(10, 0.5)  # 10 experts in half a second
    cache._drain_self_timed()

    bytes_per_expert = cache.bytes_per_expert
    assert cache.rate_tracker.observations("cpu") == 1
    assert abs(cache.rate_tracker.rate("cpu") - 20 * bytes_per_expert) < 1e-6
