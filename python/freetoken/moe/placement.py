"""Dividing a decode step's expert misses among however many executors a machine has.

The offload backend has always had two places to put a missing expert -- fetch it over the
link and compute it on the main device, or leave it in host memory and compute it on the
CPU -- and it splits them by a single ratio fixed at start-up, ``pcie / (pcie + cpu)``.
That ratio is right for exactly two executors and cannot express a third, so a machine with
another accelerator had no way to say so: its layers were assigned by hand, before the first
token, and no measurement could move them.

This module is the general form of that ratio. Every executor is described by one number --
how many bytes of expert weight it can turn into a result per second, end to end, including
whatever it has to move to get there -- and the split is the one that has them all finish at
the same moment. Two executors reduce to the old formula exactly; N executors need no new
idea, which is the point.

**Rates are end-to-end on purpose.** The main device's rate is bounded by its link, the CPU's
by memory bandwidth and cores, another accelerator's by whichever of its own two limits binds
first. Collapsing each to a single throughput is what lets them be compared at all, and it is
what the benchmark profile already reports, so nothing new has to be measured to start.

**Nothing here names a device kind.** An executor is a name and a rate. What a machine has,
how many, and which is fastest are inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecutorRate:
    """One place experts can be computed, and how fast it gets through them.

    ``bytes_per_second`` is end-to-end for a *missing* expert: the bytes it must move plus
    the compute, as one throughput. ``busy_seconds`` is work this executor is already
    committed to for this step and which the split cannot avoid -- the main device computing
    the experts it already holds, say. An executor that starts out busy is given fewer
    misses, so the finish times still line up.
    """

    name: str
    bytes_per_second: float
    busy_seconds: float = 0.0

    @property
    def usable(self) -> bool:
        return self.bytes_per_second > 0.0


@dataclass(frozen=True)
class Placement:
    """How many misses each executor takes, and what it is expected to cost."""

    counts: dict[str, int]
    makespan_seconds: float
    per_executor_seconds: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        parts = [
            f"{name}={count}" for name, count in sorted(self.counts.items()) if count
        ]
        return (
            f"{', '.join(parts) or 'nothing to place'} "
            f"(finishing together in {self.makespan_seconds * 1e3:.2f} ms)"
        )


def _finish_time(rate: ExecutorRate, count: int, bytes_per_expert: int) -> float:
    if not rate.usable:
        return float("inf")
    return rate.busy_seconds + (count * bytes_per_expert) / rate.bytes_per_second


def _continuous_shares(
    rates: list[ExecutorRate], misses: int, bytes_per_expert: int
) -> dict[str, float]:
    """Real-valued counts that make every usable executor finish at the same instant.

    Water-filling: at a common finish time ``T`` an executor can absorb
    ``(T - busy) * rate / bytes`` experts, which is zero until ``T`` passes its existing
    commitment. Total capacity rises monotonically with ``T``, so the ``T`` where it equals
    the number of misses is found by bisection -- no closed form is needed and none of the
    corner cases (an executor that never gets any, one that starts far behind) need
    special handling.
    """
    lo, hi = 0.0, 1.0
    # Grow the bracket until the fastest arrangement could absorb everything.
    while sum(
        max(0.0, (hi - r.busy_seconds) * r.bytes_per_second / bytes_per_expert)
        for r in rates
    ) < misses:
        hi *= 2.0
        if hi > 1e6:  # a second is already absurd; refuse to spin
            break
    for _ in range(200):  # bisection to well past float precision
        mid = (lo + hi) / 2.0
        total = sum(
            max(0.0, (mid - r.busy_seconds) * r.bytes_per_second / bytes_per_expert)
            for r in rates
        )
        if total < misses:
            lo = mid
        else:
            hi = mid
    return {
        r.name: max(0.0, (hi - r.busy_seconds) * r.bytes_per_second / bytes_per_expert)
        for r in rates
    }


def split_misses(
    executors: list[ExecutorRate], misses: int, bytes_per_expert: int
) -> Placement:
    """Divide ``misses`` experts among ``executors`` so they finish as close to together
    as whole experts allow.

    An expert cannot be halved, so the real-valued solution is rounded by largest
    remainder: everyone takes their whole part, and the experts left over go to whoever was
    cut hardest. That keeps the total exact -- every miss is placed exactly once, which
    matters more than the last few microseconds of balance, since a dropped miss is a wrong
    answer and a duplicated one is wasted work.

    Ordering is by name wherever remainders tie, so the same step always splits the same
    way and a difference between two runs is a real difference.
    """
    usable = [r for r in executors if r.usable]
    if not usable:
        raise ValueError("no usable executor: every rate was zero or negative")
    if misses <= 0:
        return Placement(
            counts={r.name: 0 for r in usable},
            makespan_seconds=max((r.busy_seconds for r in usable), default=0.0),
            per_executor_seconds={r.name: r.busy_seconds for r in usable},
        )

    shares = _continuous_shares(usable, misses, bytes_per_expert)
    counts = {name: int(value) for name, value in shares.items()}
    remainder = misses - sum(counts.values())
    if remainder > 0:
        by_fraction = sorted(
            usable, key=lambda r: (-(shares[r.name] - counts[r.name]), r.name)
        )
        for rate in by_fraction[:remainder]:
            counts[rate.name] += 1
        # More experts left than executors: hand the rest to the fastest, which is where
        # they cost least, rather than spreading them over executors already at capacity.
        left = misses - sum(counts.values())
        if left > 0:
            fastest = max(usable, key=lambda r: (r.bytes_per_second, r.name))
            counts[fastest.name] += left

    per_executor = {
        r.name: _finish_time(r, counts[r.name], bytes_per_expert) for r in usable
    }
    return Placement(
        counts=counts,
        makespan_seconds=max(per_executor.values()),
        per_executor_seconds=per_executor,
    )


class RateTracker:
    """Each executor's throughput, learned from the work it actually does.

    A rate measured once at start-up describes a machine that is idle and a step that is
    synthetic, and neither is the regime the split has to be right in: the executors
    contend for the same memory controller, the batch changes shape, and a device that
    throttles is slower an hour in than it was at second one. So nothing is benchmarked
    here. Every step already knows how many experts each executor was given and how long it
    took, which is a measurement of exactly the thing the split needs -- taking it costs a
    clock read, and it is taken under real contention by construction.

    Observations are smoothed exponentially, because one step is short enough that a
    scheduler hiccup can dominate it, and because the split should follow a device that is
    genuinely slowing down rather than chase noise. ``seed`` supplies a starting rate for an
    executor that has not run yet -- the benchmark profile where there is one, or simply an
    equal share, which costs one badly-split step before measurement takes over.
    """

    __slots__ = ("_rates", "_smoothing", "_observations")

    def __init__(self, seed: dict[str, float] | None = None, smoothing: float = 0.25) -> None:
        assert 0.0 < smoothing <= 1.0, smoothing
        self._rates: dict[str, float] = {k: float(v) for k, v in (seed or {}).items() if v > 0}
        self._smoothing = smoothing
        self._observations: dict[str, int] = {}

    def observe(self, name: str, experts: int, seconds: float, bytes_per_expert: int) -> None:
        """Record that ``name`` got through ``experts`` experts in ``seconds``.

        Steps that placed nothing on an executor say nothing about its speed, and a
        non-positive duration is a clock artefact rather than infinite throughput; both are
        ignored instead of poisoning the average.
        """
        if experts <= 0 or seconds <= 0.0:
            return
        rate = (experts * bytes_per_expert) / seconds
        previous = self._rates.get(name)
        self._rates[name] = (
            rate if previous is None
            else previous + self._smoothing * (rate - previous)
        )
        self._observations[name] = self._observations.get(name, 0) + 1

    def rate(self, name: str) -> float | None:
        return self._rates.get(name)

    def observations(self, name: str) -> int:
        return self._observations.get(name, 0)

    def rates(self, names: list[str], *, busy: dict[str, float] | None = None) -> list[ExecutorRate]:
        """Current rates for ``names``, as executors ready to be split across.

        A name with no rate yet and no seed is given the mean of the ones that do have one,
        so it is tried rather than starved: an executor that is never given work never
        produces an observation, and would sit unmeasured and unused for the whole run.
        """
        known = [self._rates[n] for n in names if n in self._rates]
        fallback = (sum(known) / len(known)) if known else 1.0
        return rates_from_throughputs(
            {name: self._rates.get(name, fallback) for name in names}, busy=busy
        )

    def describe(self) -> str:
        return ", ".join(
            f"{name}={rate / 1e9:.2f} GB/s"
            f"{'' if self._observations.get(name) else ' (seed)'}"
            for name, rate in sorted(self._rates.items())
        )


def rates_from_throughputs(
    throughputs: dict[str, float],
    *,
    busy: dict[str, float] | None = None,
) -> list[ExecutorRate]:
    """Build executor rates from ``{name: bytes per second}``, dropping unusable ones.

    The benchmark profile reports exactly this shape for the two executors it knows about,
    and a measured worker adds its own entry, so a machine's whole set arrives here without
    this module learning what any of them is.
    """
    busy = busy or {}
    return [
        ExecutorRate(name, float(rate), float(busy.get(name, 0.0)))
        for name, rate in sorted(throughputs.items())
        if float(rate) > 0.0
    ]


# --------------------------------------------------------------------------------------
# Deciding per miss count, for the device to look up.
#
# The split above is one ratio for a layer, applied to whatever the step brings. That is the
# right answer only when an executor's cost is proportional to its work, and for a helper it
# is not: waking a pool and handing it a layer costs the same whether it gets one expert or
# six. A ratio cannot say "give it none, or give it several", so it hands out single experts
# that cost more to hand over than they save -- and, rated from the steps where it had many,
# a helper looks fast enough to be given the lot. The device knows how many experts a layer
# is missing before it moves any; what it lacks is the answer for that number. So the host
# answers for every number at once, and the device reads the row it has.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutorCost:
    """What one executor costs a layer, as a fixed part and a part per expert.

    For the main device ``fixed_seconds`` is the work it owes whatever the placement decides
    (its GEMM) and ``per_expert_seconds`` a fetch over its link. For a helper the fixed part
    is only paid when it is given something: waking it, handing the layer over, taking the
    answer back.
    """

    name: str
    per_expert_seconds: float
    fixed_seconds: float = 0.0


def _helper_seconds(cost: ExecutorCost, count: int) -> float:
    return cost.fixed_seconds + count * cost.per_expert_seconds if count else 0.0


def plan_miss_counts(
    main: ExecutorCost,
    helpers: list[ExecutorCost],
    max_misses: int,
    min_main: list[int] | None = None,
) -> list[tuple[int, ...]]:
    """For every miss count ``m`` in ``[0, max_misses]``, how many each executor takes.

    Row ``m`` is ``(main, helper_0, helper_1, ...)`` summing to ``m``: the split with the
    shortest layer -- the slowest executor's finish -- over every integer division. Ties go
    to the arrangement that hands helpers the fewest experts, so an executor that cannot
    shorten the layer is not woken for nothing. The search is exhaustive; with a handful of
    helpers and a top-k of eight it is a few hundred candidates per row, computed on the
    host when the rates move, never on the decode path.

    ``min_main[m]`` is the least the main device fetches at ``m`` misses. A fetch is worth
    more than the layer it serves: the expert stays in the slot cache and the next step
    that routes to it is a hit, which no helper's compute can buy. The makespan alone
    cannot see that -- given a helper cheaper per expert, it hands the helper every miss,
    the cache never warms, and every route stays a miss (measured: 100 % misses, 7.7 tok/s
    where the proportional split ran at 11.7). The floor keeps the main device's share of
    the fetching; the plan decides how the rest divides.
    """
    rows: list[tuple[int, ...]] = []
    n = len(helpers)
    for misses in range(max_misses + 1):
        floor = min(misses, min_main[misses]) if min_main is not None else 0
        best_key, best = None, None

        def search(index: int, left: int, taken: tuple[int, ...]) -> None:
            nonlocal best_key, best
            if index == n:
                main_count = left
                if main_count < floor:
                    return
                span = main.fixed_seconds + main_count * main.per_expert_seconds
                for cost, count in zip(helpers, taken):
                    span = max(span, _helper_seconds(cost, count))
                key = (span, misses - main_count, taken)
                if best_key is None or key < best_key:
                    best_key, best = key, (main_count, *taken)
                return
            for count in range(left + 1):
                search(index + 1, left - count, taken + (count,))

        search(0, misses, ())
        rows.append(best)
    return rows


class CostTracker:
    """Each executor's fixed and per-expert cost, fitted from the work it reports.

    A sample is ``(tasks, experts, seconds)`` -- one layer from an executor that times each
    layer, or a window of many from one that keeps totals. The fit is least squares over the
    most recent windows, ``seconds ~ fixed * tasks + per_expert * experts``: a pool that is
    handed one expert at a time and one handed eight give different averages, and only the
    two terms together say which it will be next time. When the samples cannot separate the
    terms (every window with the same experts per task), the fixed part is taken as zero and
    the cost is the plain average -- the same answer a rate would give.
    """

    __slots__ = ("_samples", "_window")

    def __init__(self, window: int = 64) -> None:
        self._window = window
        self._samples: dict[str, list[tuple[float, float, float]]] = {}

    def observe(self, name: str, tasks: int, experts: int, seconds: float) -> None:
        if tasks <= 0 or experts <= 0 or seconds <= 0.0:
            return
        bucket = self._samples.setdefault(name, [])
        bucket.append((float(tasks), float(experts), float(seconds)))
        if len(bucket) > self._window:
            del bucket[: len(bucket) - self._window]

    def samples(self, name: str) -> int:
        return len(self._samples.get(name, ()))

    def cost(self, name: str) -> tuple[float, float] | None:
        """``(fixed_seconds, per_expert_seconds)``, or None before any sample."""
        bucket = self._samples.get(name)
        if not bucket:
            return None
        stt = sum(t * t for t, _, _ in bucket)
        see = sum(e * e for _, e, _ in bucket)
        ste = sum(t * e for t, e, _ in bucket)
        sts = sum(t * s for t, _, s in bucket)
        ses = sum(e * s for _, e, s in bucket)
        det = stt * see - ste * ste
        if det > 1e-9 * stt * see:
            fixed = (sts * see - ses * ste) / det
            per_expert = (stt * ses - ste * sts) / det
            if fixed >= 0.0 and per_expert > 0.0:
                return fixed, per_expert
        total_e = sum(e for _, e, _ in bucket)
        return 0.0, sum(s for _, _, s in bucket) / total_e


__all__ = [
    "CostTracker",
    "ExecutorCost",
    "ExecutorRate",
    "Placement",
    "RateTracker",
    "plan_miss_counts",
    "rates_from_throughputs",
    "split_misses",
]
