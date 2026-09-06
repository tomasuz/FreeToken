"""Splitting a step's expert misses across however many executors exist.

The property that matters is not elegance of the division but that it is a division: every
miss placed exactly once, because a dropped one is a wrong answer and a duplicated one is
wasted work. After that, the split should make the executors finish together, and should
reduce to the old two-way ratio it replaces so the change is a generalisation rather than a
new behaviour nobody asked for.
"""

from __future__ import annotations

import pytest

from freetoken.moe.placement import (
    ExecutorRate,
    rates_from_throughputs,
    split_misses,
)

BYTES = 3_342_336  # one expert of the model this was built against


def test_every_miss_is_placed_exactly_once():
    executors = [ExecutorRate("gpu", 13e9), ExecutorRate("cpu", 2e9), ExecutorRate("igpu", 5e9)]

    for misses in range(0, 40):
        placement = split_misses(executors, misses, BYTES)
        assert sum(placement.counts.values()) == misses, misses
        assert all(count >= 0 for count in placement.counts.values())


def test_a_lone_executor_takes_everything():
    placement = split_misses([ExecutorRate("gpu", 13e9)], 8, BYTES)

    assert placement.counts == {"gpu": 8}


def test_two_executors_reproduce_the_ratio_this_replaces():
    """The old split was pcie / (pcie + cpu); the general form must agree with it."""
    pcie, cpu = 12e9, 4e9
    misses = 400  # large enough that rounding is not the story

    placement = split_misses(
        [ExecutorRate("gpu", pcie), ExecutorRate("cpu", cpu)], misses, BYTES
    )

    expected_gpu = misses * pcie / (pcie + cpu)
    assert placement.counts["gpu"] == pytest.approx(expected_gpu, abs=1)


def test_a_third_executor_takes_work_off_both_others():
    """The point of the exercise: another device must be able to earn a share."""
    two = split_misses([ExecutorRate("gpu", 12e9), ExecutorRate("cpu", 4e9)], 64, BYTES)
    three = split_misses(
        [ExecutorRate("gpu", 12e9), ExecutorRate("cpu", 4e9), ExecutorRate("igpu", 8e9)],
        64,
        BYTES,
    )

    assert three.counts["igpu"] > 0
    assert three.counts["gpu"] < two.counts["gpu"]
    assert three.counts["cpu"] < two.counts["cpu"]
    assert three.makespan_seconds < two.makespan_seconds  # that is why it was added


def test_executors_finish_within_one_expert_of_each_other():
    """Balance is the whole objective; whole experts are the only reason to miss it."""
    executors = [ExecutorRate("gpu", 13e9), ExecutorRate("cpu", 2.5e9), ExecutorRate("igpu", 6e9)]

    placement = split_misses(executors, 120, BYTES)

    times = placement.per_executor_seconds
    slowest_expert = BYTES / min(r.bytes_per_second for r in executors)
    assert max(times.values()) - min(times.values()) <= slowest_expert


def test_an_executor_already_busy_is_given_less():
    """The main device also computes the experts it already holds; that time is real."""
    idle = split_misses(
        [ExecutorRate("gpu", 10e9), ExecutorRate("cpu", 10e9)], 40, BYTES
    )
    busy = split_misses(
        [ExecutorRate("gpu", 10e9, busy_seconds=0.005), ExecutorRate("cpu", 10e9)],
        40,
        BYTES,
    )

    assert busy.counts["gpu"] < idle.counts["gpu"]
    assert busy.counts["cpu"] > idle.counts["cpu"]
    assert sum(busy.counts.values()) == 40


def test_an_executor_too_busy_to_help_is_given_nothing():
    busy = ExecutorRate("gpu", 10e9, busy_seconds=10.0)
    placement = split_misses([busy, ExecutorRate("cpu", 10e9)], 8, BYTES)

    assert placement.counts["gpu"] == 0
    assert placement.counts["cpu"] == 8


def test_a_much_slower_executor_still_gets_a_share_but_a_small_one():
    placement = split_misses(
        [ExecutorRate("fast", 100e9), ExecutorRate("slow", 1e9)], 202, BYTES
    )

    assert placement.counts["slow"] >= 1
    assert placement.counts["fast"] > 50 * placement.counts["slow"]


def test_nothing_to_place_is_not_an_error():
    placement = split_misses([ExecutorRate("gpu", 13e9)], 0, BYTES)

    assert placement.counts == {"gpu": 0}


def test_unusable_executors_are_dropped_not_handed_work():
    placement = split_misses(
        [ExecutorRate("gpu", 13e9), ExecutorRate("broken", 0.0)], 8, BYTES
    )

    assert "broken" not in placement.counts
    assert placement.counts["gpu"] == 8


def test_no_usable_executor_is_refused_rather_than_silently_dropping_work():
    with pytest.raises(ValueError, match="no usable executor"):
        split_misses([ExecutorRate("broken", 0.0)], 8, BYTES)


def test_the_split_is_the_same_every_time_it_is_asked():
    """A run-to-run difference should mean something changed, not that ties broke elsewhere."""
    executors = [ExecutorRate("a", 5e9), ExecutorRate("b", 5e9), ExecutorRate("c", 5e9)]

    first = split_misses(executors, 7, BYTES)
    for _ in range(5):
        assert split_misses(executors, 7, BYTES).counts == first.counts


def test_throughputs_become_rates_and_zero_ones_are_dropped():
    rates = rates_from_throughputs(
        {"gpu": 13e9, "cpu": 2e9, "absent": 0.0}, busy={"gpu": 0.002}
    )

    assert [r.name for r in rates] == ["cpu", "gpu"]
    assert next(r for r in rates if r.name == "gpu").busy_seconds == 0.002


def test_description_names_who_got_what():
    placement = split_misses(
        [ExecutorRate("gpu", 12e9), ExecutorRate("cpu", 4e9)], 16, BYTES
    )

    text = placement.describe()
    assert "gpu=" in text and "cpu=" in text and "ms" in text


# ---------------------------------------------------------------------------
# rates learned from real work
# ---------------------------------------------------------------------------


def test_a_tracker_converges_on_the_rate_it_keeps_seeing():
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker()
    for _ in range(50):
        tracker.observe("gpu", experts=10, seconds=10 * BYTES / 12e9, bytes_per_expert=BYTES)

    assert tracker.rate("gpu") == pytest.approx(12e9, rel=0.01)


def test_a_tracker_follows_an_executor_that_slows_down():
    """A device that throttles must lose its share, which a start-up benchmark cannot see."""
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker()
    for _ in range(40):
        tracker.observe("gpu", 10, 10 * BYTES / 12e9, BYTES)
    fast = tracker.rate("gpu")
    for _ in range(40):
        tracker.observe("gpu", 10, 10 * BYTES / 3e9, BYTES)

    assert tracker.rate("gpu") < fast / 3


def test_a_step_that_placed_nothing_says_nothing_about_speed():
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker(seed={"cpu": 2e9})
    tracker.observe("cpu", experts=0, seconds=0.004, bytes_per_expert=BYTES)

    assert tracker.rate("cpu") == 2e9
    assert tracker.observations("cpu") == 0


def test_a_clock_artefact_is_ignored_rather_than_read_as_infinite_speed():
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker(seed={"cpu": 2e9})
    tracker.observe("cpu", experts=4, seconds=0.0, bytes_per_expert=BYTES)
    tracker.observe("cpu", experts=4, seconds=-1e-9, bytes_per_expert=BYTES)

    assert tracker.rate("cpu") == 2e9


def test_an_unmeasured_executor_is_tried_rather_than_starved():
    """An executor given no work produces no observation, so it must not need one first."""
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker(seed={"gpu": 12e9, "cpu": 4e9})

    rates = tracker.rates(["gpu", "cpu", "igpu"])

    igpu = next(r for r in rates if r.name == "igpu")
    assert igpu.bytes_per_second > 0
    placement = split_misses(rates, 32, BYTES)
    assert placement.counts["igpu"] > 0


def test_seeded_rates_are_marked_as_such_until_something_is_measured():
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker(seed={"gpu": 12e9})
    assert "(seed)" in tracker.describe()

    tracker.observe("gpu", 8, 8 * BYTES / 12e9, BYTES)
    assert "(seed)" not in tracker.describe()


def test_measured_rates_drive_the_split_towards_whoever_is_actually_faster():
    """The closed loop: observe, then split by what was observed rather than assumed."""
    from freetoken.moe.placement import RateTracker

    tracker = RateTracker(seed={"gpu": 8e9, "igpu": 8e9})
    before = split_misses(tracker.rates(["gpu", "igpu"]), 40, BYTES).counts
    assert before["gpu"] == before["igpu"]  # seeded equal, split equal

    for _ in range(40):  # the iGPU turns out to be a third the speed
        tracker.observe("gpu", 10, 10 * BYTES / 9e9, BYTES)
        tracker.observe("igpu", 10, 10 * BYTES / 3e9, BYTES)
    after = split_misses(tracker.rates(["gpu", "igpu"]), 40, BYTES).counts

    assert after["gpu"] > before["gpu"]
    assert after["igpu"] < before["igpu"]
    assert sum(after.values()) == 40
