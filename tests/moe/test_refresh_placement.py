"""Rewriting the placement a captured decode reads, once per step.

A replay is not launched until the host has written the step's division, so the division is
host time on every token. It depends on the executors taking part and their rates, not on the
layer, which is why it is computed once per set of helpers. What has to hold is that nothing
a replay reads changes because of that: every layer's owner map and fetch fraction must be
what dividing that layer on its own would have written.
"""

from __future__ import annotations

import torch

from freetoken.layers.moe import _fill_owner_map
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.placement import RateTracker

SHAPE = (4, 8)  # a batch of four, top-8: sixteen positions per helper is enough to see a split


class _Executor:
    """An accelerator that serves some layers and has measured nothing of its own."""

    def __init__(self, device_index, layers):
        self.device_index = device_index
        self._layers = frozenset(layers)

    def serves(self, layer_id):
        return layer_id in self._layers

    def take_samples(self):
        return []


def _cache():
    cache = OffloadMoeCache(
        num_layers=6, num_experts=8, cache_size=8,
        device=torch.device("cpu"), quant_format="q4_0", prefill_overlap=False,
    )
    # Layers 0-1 have gpu1, 2-3 have gpu1 and gpu2, 4 has gpu2, 5 has nobody.
    cache.device_executors = [_Executor(1, {0, 1, 2, 3}), _Executor(2, {2, 3, 4})]
    cache.rate_tracker = RateTracker({"gpu": 12.0, "gpu1": 1.8, "gpu2": 3.0})
    for layer_id in range(6):
        cache.owner_map(layer_id, SHAPE, [("unmeasured", 1.0)])
    return cache


def _divided_alone(cache, layer_id):
    """What the placement was before it was shared: this layer divided on its own."""
    helpers = list(cache.split_helpers(layer_id))
    shares = OffloadMoeCache.split_shares(cache, layer_id, helpers)
    owner = torch.zeros(SHAPE, dtype=torch.int32)
    _fill_owner_map(owner, [(name, shares.get(name, 0.0)) for name in helpers])
    fraction = min(1 << 16, max(0, round(shares.get("gpu", 1.0) * (1 << 16))))
    return owner, fraction


def _owner(cache, layer_id):
    return cache._owner_maps[(layer_id, SHAPE)][0]


def test_the_division_is_computed_once_per_set_of_helpers():
    cache = _cache()
    calls = []
    divide = cache.split_shares
    cache.split_shares = lambda layer_id, names: calls.append(tuple(names)) or divide(layer_id, names)

    cache.refresh_placement()

    assert sorted(calls) == [("gpu1",), ("gpu1", "gpu2"), ("gpu2",)]


def test_every_layer_reads_what_dividing_it_alone_would_have_written():
    cache = _cache()
    untouched = int(cache._fetch_frac_host[5])

    cache.refresh_placement()

    for layer_id in range(5):
        owner, fraction = _divided_alone(cache, layer_id)
        assert torch.equal(_owner(cache, layer_id), owner), layer_id
        assert int(cache._fetch_frac_host[layer_id]) == fraction, layer_id
    assert 0 < int(cache._fetch_frac_host[0]) < 1 << 16, "the rates should split, not saturate"
    assert int(cache._fetch_frac_host[5]) == untouched, "a layer with no helper is left alone"


def test_a_rate_that_moves_between_steps_moves_the_next_division():
    """Shared within a step, never carried to the next: the rates are re-read every time."""
    cache = _cache()
    cache.refresh_placement()
    before = int(cache._fetch_frac_host[0])

    cache.rate_tracker = RateTracker({"gpu": 12.0, "gpu1": 12.0, "gpu2": 3.0})
    cache.refresh_placement()

    assert int(cache._fetch_frac_host[0]) < before
    for layer_id in range(5):
        owner, fraction = _divided_alone(cache, layer_id)
        assert torch.equal(_owner(cache, layer_id), owner), layer_id
        assert int(cache._fetch_frac_host[layer_id]) == fraction, layer_id
