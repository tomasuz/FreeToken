"""The slot-cache side of the CPU-assisted MoE decode (``OffloadMoeCache.assist_split`` /
``assist_plan`` / ``assist_join``).

Checked on a small cache: which routes the GPU takes, which misses are fetched now or
admitted in the background, that admitted slots hold their expert's bytes, and that a
step too big for the LRU plan (a large decode batch) is served without one instead of
failing.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

LAYERS, EXPERTS, SLOTS = 4, 16, 24
FEATS = [8192, 512, 256, 4096, 512, 256]


def _cache():
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=LAYERS, num_experts=EXPERTS, cache_size=SLOTS,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    sources = {
        name: list(torch.randint(0, 256, (LAYERS * EXPERTS, feat), dtype=torch.uint8, device=dev).split(EXPERTS))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)
    return cache


def _make_resident(cache, layer, experts):
    ids = torch.tensor([experts], dtype=torch.int32, device="cuda")
    cache.ensure_experts(layer, ids)
    cache.copy_missing()
    torch.cuda.synchronize()


def _slots(cache, layer, ids):
    flat = (ids.to(torch.int64) + layer * EXPERTS).reshape(-1)
    return cache.slot_for_id.view(-1).index_select(0, flat).view(ids.shape)


def _holds(cache, layer, expert, slot):
    """Whether cache slot ``slot`` holds the bytes of ``expert`` of ``layer``, every bank."""
    for per_layer, bank in cache.banks:
        if not torch.equal(bank[slot], per_layer[layer][expert]):
            return False
    return True


def test_small_step_admits_capped_misses():
    cache = _cache()
    layer = 1
    _make_resident(cache, layer, [0, 1, 2, 3])
    ids = torch.tensor([[0, 5, 1, 6], [2, 7, 3, 8]], dtype=torch.int32, device="cuda")
    gpu = cache.assist_split(layer, ids)
    assert gpu.tolist() == [[True, False, True, False], [True, False, True, False]]
    before = _slots(cache, layer, ids)
    slots = cache.assist_plan(layer, ids, gpu, admit_cap=1, gpu_fetches=False)
    cache.assist_join()
    torch.cuda.synchronize()
    # the GPU routes read their resident slots, unchanged by the plan
    assert torch.equal(slots[gpu], before[gpu])
    # exactly one miss admitted (the first in route order: expert 5), its bytes copied in
    after = _slots(cache, layer, ids)
    admitted = [e for e in (5, 6, 7, 8) if int(after[ids == e][0]) >= 0]
    assert admitted == [5]
    assert _holds(cache, layer, 5, int(after[ids == 5][0]))


def test_cpu_budget_fetches_the_rest_now():
    cache = _cache()
    layer = 2
    _make_resident(cache, layer, [0])
    ids = torch.tensor([[0, 4, 5, 6, 7]], dtype=torch.int32, device="cuda")
    gpu = cache.assist_split(layer, ids, cpu_max=2)
    # misses 4 and 5 go to the CPU, 6 and 7 past the budget are the GPU's
    assert gpu.tolist() == [[True, False, False, True, True]]
    slots = cache.assist_plan(layer, ids, gpu, admit_cap=0, gpu_fetches=True)
    torch.cuda.synchronize()  # the GPU's own fetch is on this stream: landed before its GEMM
    for route in (0, 3, 4):
        assert _holds(cache, layer, int(ids[0, route]), int(slots[0, route]))
    cache.assist_join()
    torch.cuda.synchronize()
    for expert in (4, 5):  # admitted in the background (no cap)
        slot = int(_slots(cache, layer, torch.tensor([[expert]], device="cuda"))[0, 0])
        assert slot >= 0 and _holds(cache, layer, expert, slot)


def test_step_too_big_for_the_plan_is_served_without_one():
    cache = _cache()
    layer = 3
    _make_resident(cache, layer, [0, 1])
    routes = SLOTS // 2 + 1  # more than half the slot region
    ids = torch.tensor([[e % EXPERTS for e in range(routes)]], dtype=torch.int32, device="cuda")
    assert not cache.assist_plannable(layer, routes)
    gpu = cache.assist_split(layer, ids, cpu_max=2)  # the budget cannot apply: nothing fetched
    assert gpu.tolist() == [[e % EXPERTS in (0, 1) for e in range(routes)]]
    resident_before = cache.slot_for_id.clone()
    slots = cache.assist_plan(layer, ids, gpu, admit_cap=1, gpu_fetches=True)
    cache.assist_join()
    torch.cuda.synchronize()
    assert torch.equal(cache.slot_for_id, resident_before)  # nothing admitted, nothing evicted
    for route in range(routes):
        if bool(gpu[0, route]):
            assert _holds(cache, layer, int(ids[0, route]), int(slots[0, route]))


def test_plan_buffers_grow_past_the_default_capacity():
    cache = _cache()
    big = OffloadMoeCache._PF_CAPACITY + 8
    assert cache._assist_buffers(big)[0].shape[-1] >= big
