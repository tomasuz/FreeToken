"""Assigning expert layers to a worker on another device.

Placement is the part that can go wrong quietly: a layer handed to two owners, or one whose
weights the cache still tries to move after somebody else took them. The failure would be a
wrong answer or a fault deep in a copy, so it is pinned down here rather than left to an
end-to-end run.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.engine import (
    _middle_first,
    _parse_per_device,
    _resolve_worker_layers,
)


class _Cfg:
    moe_backend = "offload"

    def __init__(self, layers=None, env=None):
        self.moe_worker_layers = layers
        self.moe_worker_env = env


def test_per_device_spec_parsing():
    assert _parse_per_device("1:0-3", "x") == {1: "0-3"}
    assert _parse_per_device("1:a;2:b", "x") == {1: "a", 2: "b"}
    assert _parse_per_device(None, "x") == {}
    assert _parse_per_device("", "x") == {}
    with pytest.raises(ValueError, match="expected"):
        _parse_per_device("nocolon", "x")
    with pytest.raises(ValueError, match="not a device index"):
        _parse_per_device("gpu1:0-3", "x")
    with pytest.raises(ValueError, match="twice"):
        _parse_per_device("1:a;1:b", "x")


def test_explicit_ids_are_honoured():
    assert _resolve_worker_layers(_Cfg("1:3,7,11"), 30) == {1: frozenset({3, 7, 11})}


def test_a_count_is_taken_from_the_middle():
    """The resident tier takes the ends; a worker takes the middle, and for the same reason.

    Decode miss rates are U-shaped, so the middle is what the offload cache serves most
    cheaply -- and therefore what costs least to hand away.
    """
    got = _resolve_worker_layers(_Cfg("1:4"), 30)[1]
    assert len(got) == 4
    assert all(8 <= i <= 21 for i in got), sorted(got)


def test_middle_first_ordering():
    assert _middle_first(5)[:3] == [2, 1, 3]
    assert sorted(_middle_first(7)) == list(range(7))


def test_two_devices_do_not_share_a_layer():
    got = _resolve_worker_layers(_Cfg("1:3;2:3"), 30)
    all_ids = [i for ids in got.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)) == 6


def test_overlapping_explicit_ids_are_refused():
    with pytest.raises(ValueError, match="more than one device"):
        _resolve_worker_layers(_Cfg("1:5,6;2:6,7"), 30)


def test_worker_layers_are_offload_only():
    cfg = _Cfg("1:4")
    cfg.moe_backend = "fused"
    assert _resolve_worker_layers(cfg, 30) == {}


def _cache(num_layers=4, num_experts=8):
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=num_experts,
        device=torch.device("cpu"), quant_format="q4_0", prefill_overlap=False,
    )


def test_worker_layers_are_excluded_from_movement():
    """A layer someone else owns must not be read or copied by this cache."""
    cache = _cache()
    cache.set_worker_executors({1: object(), 2: object()})
    assert cache.is_worker_layer(1) and cache.is_worker_layer(2)
    assert not cache.is_worker_layer(0)
    assert cache._skips_movement(1) and not cache._skips_movement(0)

    sources = {
        n: [torch.zeros(8, 32, dtype=torch.uint8) for _ in range(4)]
        for n in ("gate_up", "down")
    }
    # a worker layer's source may be anything by now; validation must not touch it
    sources["gate_up"][1] = torch.zeros(8, 64, dtype=torch.uint8)[:, ::2]
    cache.set_bank_sources(sources)  # must not raise


def test_a_layer_cannot_be_owned_twice():
    cache = _cache()
    cache.cpu_layer_ids = frozenset({1})
    with pytest.raises(AssertionError, match="worker and CPU decode"):
        cache.set_worker_executors({1: object()})

    cache2 = _cache()
    resident = {n: {2: torch.zeros(8, 32, dtype=torch.uint8)} for n in ("gate_up", "down")}
    cache2.set_resident_banks(resident, frozenset({2}))
    with pytest.raises(AssertionError, match="worker-served and VRAM-resident"):
        cache2.set_worker_executors({2: object()})
