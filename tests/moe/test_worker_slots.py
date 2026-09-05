"""The worker's LRU slot cache over expert rows.

What matters here is not that rows arrive on the device -- a copy is a copy -- but that the
remapped ids point at the right ones. A cache that silently hands back the wrong slot
computes with the wrong expert and produces plausible tokens, so every test that checks
placement also checks the bytes behind it.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.worker_slots import WorkerSlotCache

E, ROW = 8, 4  # small enough to reason about, wide enough to catch a row mix-up
CPU = torch.device("cpu")


def banks(num_experts: int = E) -> dict[str, torch.Tensor]:
    """Expert rows whose contents identify their own expert id."""
    gate_up = torch.arange(num_experts, dtype=torch.uint8).repeat_interleave(ROW).reshape(num_experts, ROW)
    down = gate_up + 100
    return {"gate_up": gate_up, "down": down}


def ids(*rows: list[int]) -> torch.Tensor:
    return torch.tensor(list(rows), dtype=torch.int32)


def test_remapped_ids_point_at_the_requested_experts():
    """The whole cache is wrong if a slot holds an expert other than the one asked for."""
    host = banks()
    cache = WorkerSlotCache(host, slots=4, device=CPU)

    out = cache.ensure(ids([3, 5]))

    for token, row in enumerate(out.tolist()):
        for k, slot in enumerate(row):
            expert = int(ids([3, 5])[token][k])
            assert torch.equal(cache.dev["gate_up"][slot], host["gate_up"][expert])
            assert torch.equal(cache.dev["down"][slot], host["down"][expert])


def test_a_second_look_at_the_same_expert_is_a_hit():
    host = banks()
    cache = WorkerSlotCache(host, slots=4, device=CPU)

    first = cache.ensure(ids([1, 2]))
    assert cache.misses == 2 and cache.hits == 0
    second = cache.ensure(ids([1, 2]))

    assert torch.equal(first, second)  # same experts, same slots
    assert cache.hits == 2
    assert cache.fills == 2  # no refill on a hit


def test_least_recently_used_expert_is_the_one_evicted():
    host = banks()
    cache = WorkerSlotCache(host, slots=2, device=CPU)

    cache.ensure(ids([0]))
    cache.ensure(ids([1]))
    cache.ensure(ids([0]))  # 0 is now newer than 1
    cache.ensure(ids([2]))  # must evict 1, not 0

    slot_for_0 = cache.ensure(ids([0]))[0][0].item()
    assert torch.equal(cache.dev["gate_up"][slot_for_0], host["gate_up"][0])
    assert cache.misses == 3  # 0, 1, 2 -- and 0 was still resident at the end


def test_an_expert_filled_this_step_is_not_evicted_by_a_later_one():
    """Within a step every routed expert must survive: the same launch reads them all."""
    host = banks()
    cache = WorkerSlotCache(host, slots=3, device=CPU)

    out = cache.ensure(ids([4, 5, 6]))

    slots = sorted(s for row in out.tolist() for s in row)
    assert slots == [0, 1, 2], "three distinct experts must occupy three distinct slots"
    for expert, slot in zip((4, 5, 6), out[0].tolist()):
        assert torch.equal(cache.dev["gate_up"][slot], host["gate_up"][expert])


def test_one_token_wider_than_the_cache_is_refused_with_the_fix_in_the_message():
    """A single token's experts are one launch; there is no splitting that saves this."""
    cache = WorkerSlotCache(banks(), slots=2, device=CPU)

    with pytest.raises(RuntimeError, match="moe-worker-slots"):
        cache.partition(ids([0, 1, 2]))


def test_a_step_wider_than_the_cache_is_split_rather_than_refused():
    """Cache size must be a memory choice, not a cap on how wide a step may be."""
    cache = WorkerSlotCache(banks(), slots=4, device=CPU)

    ranges = cache.partition(ids([0, 1], [2, 3], [4, 5], [6, 7]))

    assert ranges == [(0, 2), (2, 4)]  # two tokens per launch at four slots
    assert [hi - lo for lo, hi in ranges] == [2, 2]


def test_a_step_that_fits_is_one_range_and_no_splitting():
    cache = WorkerSlotCache(banks(), slots=8, device=CPU)

    assert cache.partition(ids([0, 1], [2, 3], [0, 1])) == [(0, 3)]


def test_partition_ranges_cover_every_token_exactly_once():
    """A dropped or repeated token would silently corrupt the layer's output."""
    cache = WorkerSlotCache(banks(), slots=3, device=CPU)
    rows = ids([0, 1], [2, 3], [4, 5], [6, 7], [0, 7])

    ranges = cache.partition(rows)

    covered = [t for lo, hi in ranges for t in range(lo, hi)]
    assert covered == list(range(rows.shape[0]))


def test_each_partitioned_range_can_actually_be_made_resident():
    """Splitting is only correct if every range it produces then fits in the cache."""
    cache = WorkerSlotCache(banks(), slots=4, device=CPU)
    rows = ids([0, 1], [2, 3], [4, 5], [6, 7])

    for lo, hi in cache.partition(rows):
        remapped = cache.ensure(rows[lo:hi])  # must not raise
        for token, row in enumerate(remapped.tolist()):
            for k, slot in enumerate(row):
                expert = int(rows[lo + token][k])
                assert torch.equal(cache.dev["gate_up"][slot], banks()["gate_up"][expert])


def test_slots_are_capped_at_the_expert_count():
    """Asking for more slots than there are experts wastes memory on rows that cannot fill."""
    cache = WorkerSlotCache(banks(), slots=999, device=CPU)

    assert cache.slots == E
    assert cache.holds_everything


def test_a_full_size_cache_never_evicts():
    host = banks()
    cache = WorkerSlotCache(host, slots=E, device=CPU)

    for expert in range(E):
        cache.ensure(ids([expert]))
    for expert in range(E):
        slot = cache.ensure(ids([expert]))[0][0].item()
        assert torch.equal(cache.dev["gate_up"][slot], host["gate_up"][expert])

    assert cache.fills == E  # each expert arrived exactly once
    assert cache.hits == E   # and the second pass found every one of them


def test_banks_must_agree_on_how_many_experts_there_are():
    host = banks()
    host["down"] = host["down"][:-1]

    with pytest.raises(AssertionError, match="expert count"):
        WorkerSlotCache(host, slots=2, device=CPU)


def test_stats_report_what_the_parent_logs():
    cache = WorkerSlotCache(banks(), slots=4, device=CPU)

    cache.ensure(ids([1, 2]))
    cache.ensure(ids([2, 3]))

    hits, misses = cache.stats()
    assert (hits, misses) == (1, 3)  # 2 was already there; 1, 2 and 3 arrived
