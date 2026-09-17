"""Placement by miss count: the host solves every count, the device reads the one it has.

The plan has to be a division first -- every miss placed exactly once, the helpers' shares
never more than there are -- and after that it has to be the plan a helper's fixed cost
implies: nothing handed over when handing over costs more than it saves, and more handed
over as the layer's misses grow. The device side then has to deal the overflow routes out by
those counts, each route to exactly one executor.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.placement import CostTracker, ExecutorCost, plan_miss_counts

MS = 1e-3
# GLM-4.5-Air on tm: a fetch over PCIe 3.0 x16, the GEMM this device owes regardless, and a
# pool that is cheap per expert but costs a wake-up to hand anything to
GPU = ExecutorCost("gpu", per_expert_seconds=0.72 * MS, fixed_seconds=0.34 * MS)
CPU = ExecutorCost("cpu", per_expert_seconds=0.30 * MS, fixed_seconds=0.60 * MS)
IGPU = ExecutorCost("gpu1", per_expert_seconds=0.82 * MS, fixed_seconds=0.40 * MS)


def _span(row, main, helpers):
    span = main.fixed_seconds + row[0] * main.per_expert_seconds
    for cost, count in zip(helpers, row[1:]):
        if count:
            span = max(span, cost.fixed_seconds + count * cost.per_expert_seconds)
    return span


@pytest.mark.parametrize("helpers", [[CPU], [IGPU], [CPU, IGPU]])
def test_every_row_is_a_division_and_never_worse_than_fetching_alone(helpers):
    rows = plan_miss_counts(GPU, helpers, 8)

    assert len(rows) == 9
    assert rows[0] == (0,) * (1 + len(helpers))
    for misses, row in enumerate(rows):
        assert sum(row) == misses and min(row) >= 0
        alone = (misses,) + (0,) * len(helpers)
        assert _span(row, GPU, helpers) <= _span(alone, GPU, helpers) + 1e-12


def test_a_helper_that_cannot_shorten_the_layer_is_never_woken():
    # one expert costs it more than this device's worst layer (8 fetches + GEMM, 6.1 ms)
    slow = ExecutorCost("cpu", per_expert_seconds=10 * MS, fixed_seconds=1 * MS)

    assert all(row[1] == 0 for row in plan_miss_counts(GPU, [slow], 8))


def test_a_fixed_cost_holds_the_helper_back_until_the_misses_pay_for_it():
    expensive_wakeup = ExecutorCost("cpu", per_expert_seconds=0.1 * MS, fixed_seconds=3 * MS)
    rows = plan_miss_counts(GPU, [expensive_wakeup], 8)

    used = [misses for misses, row in enumerate(rows) if row[1]]
    assert used and min(used) >= 4  # 3 ms of wake-up is worth it only once fetching costs more
    assert all(rows[m][1] for m in range(min(used), 9))


def test_cost_fit_separates_the_fixed_part_from_the_part_per_expert():
    tracker = CostTracker()
    for experts in (1, 2, 4, 8, 3, 5):
        tracker.observe("cpu", 1, experts, 0.6 * MS + experts * 0.3 * MS)

    fixed, per_expert = tracker.cost("cpu")
    assert fixed == pytest.approx(0.6 * MS)
    assert per_expert == pytest.approx(0.3 * MS)


def test_cost_fit_falls_back_to_the_average_when_the_terms_cannot_be_told_apart():
    tracker = CostTracker()
    tracker.observe("cpu", 10, 30, 0.03)
    tracker.observe("cpu", 20, 60, 0.06)  # every window three experts per task

    assert tracker.cost("cpu") == (0.0, pytest.approx(0.001))
    assert tracker.cost("never-seen") is None


def _counts_cache(names):
    cache = OffloadMoeCache(
        num_layers=2, num_experts=32, cache_size=40, device=torch.device("cpu"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=32,
    )
    cache.placement_counts = True
    cache.ensure_count_tables(8, len(names))
    return cache


def test_overflow_routes_are_dealt_by_the_counts_each_exactly_once():
    names = ["cpu", "gpu1"]
    cache = _counts_cache(names)
    rows = plan_miss_counts(GPU, [CPU, IGPU], 8)
    cache._write_count_rows([1], tuple(names), rows)

    for misses in range(9):
        row = rows[misses]
        overflow = torch.zeros(1, 8, dtype=torch.bool)
        positions = torch.randperm(8)[: misses - row[0]]
        overflow.view(-1)[positions] = True
        cache.num_missing_full.fill_(misses)

        assignment = cache.assign_overflow_counts(1, overflow, names)

        owned = sum(mask.to(torch.int32) for mask in assignment.values())
        assert torch.equal(owned.bool(), overflow) and int(owned.max()) <= 1, misses
        assert [int(assignment[name].sum()) for name in names] == list(row[1:]), misses


def test_until_a_plan_is_written_the_device_fetches_everything():
    cache = _counts_cache(["cpu"])

    assert cache._fetch_table_host[0].tolist() == list(range(9))
    assert int(cache._helper_bounds_host.abs().sum()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_table_fetch_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k = 32, 40, 8

    def make():
        cache = OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts,
        )
        cache.placement_counts = True
        cache.ensure_count_tables(top_k, 1)
        cache._write_count_rows([0], ("cpu",), [(max(0, m - 2), min(m, 2)) for m in range(9)])
        return cache

    gpu, ref = make(), make()
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == max(0, missing - 2), step
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
