"""Resident (VRAM-only) expert layers and file-backed host banks.

The invariant under test is the point of the feature: a resident layer's experts exist in
exactly ONE place. So these check not just that the device banks are readable, but that the
host side really let go -- the host bank is released, the copy plan carries a null source
pointer for it, and no movement call is made on its behalf.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.host_banks import (
    HostBank,
    PinPipeline,
    bank_backing_default,
    resident_upload,
)


# ---------------------------------------------------------------------------
# file-backed banks
# ---------------------------------------------------------------------------


def test_file_bank_roundtrip(tmp_path):
    """A file-backed bank behaves exactly like the anonymous one for its consumers."""
    bank = HostBank((4, 8), torch.uint8, backing="file")
    assert bank.tensor.shape == (4, 8)
    assert bank.addr % 4096 == 0, "direct-IO readers need page alignment"
    assert bank._spill is not None

    bank.tensor.copy_(torch.arange(32, dtype=torch.uint8).reshape(4, 8))
    assert torch.equal(bank.tensor.reshape(-1), torch.arange(32, dtype=torch.uint8))


def test_file_bank_release_returns_blocks(tmp_path):
    """release() must give the bytes back to the filesystem, not just drop the RAM copy.

    A plain MADV_DONTNEED would leave the spill file fully allocated -- the disk equivalent
    of the RAM leak the resident tier exists to avoid.
    """
    os.environ["FREETOKEN_BANK_SPILL_DIR"] = str(tmp_path)
    try:
        assert bank_backing_default() == "file"
        bank = HostBank((1024, 4096), torch.uint8)  # 4 MiB, > one block
        bank.tensor.fill_(7)
        fd = bank._spill.fileno()
        allocated_before = os.fstat(fd).st_blocks
        assert allocated_before > 0, "written pages should have allocated file blocks"

        bank.release()
        assert os.fstat(fd).st_blocks < allocated_before
    finally:
        del os.environ["FREETOKEN_BANK_SPILL_DIR"]


def test_spill_dir_env_drives_default(tmp_path):
    assert bank_backing_default() == "mmap"
    os.environ["FREETOKEN_BANK_SPILL_DIR"] = str(tmp_path)
    try:
        assert bank_backing_default() == "file"
    finally:
        del os.environ["FREETOKEN_BANK_SPILL_DIR"]
    assert bank_backing_default() == "mmap"


# ---------------------------------------------------------------------------
# resident uploader
# ---------------------------------------------------------------------------


class _FakeUploader:
    """Records the (layer, banks) the sink routes to it, without touching a device."""

    def __init__(self, layers):
        self.layers = frozenset(layers)
        self.seen = {}

    def claims(self, layer_id):
        return layer_id in self.layers

    def upload(self, layer_id, banks):
        self.seen[layer_id] = {n: b.tensor.clone() for n, b in banks.items()}


def test_pin_pipeline_routes_claimed_layers_to_upload_and_releases():
    """A claimed layer is uploaded and released; an unclaimed one is settled normally."""
    banks = {
        name: [HostBank((2, 16), torch.uint8) for _ in range(3)]
        for name in ("gate_up", "down")
    }
    for name, per_layer in banks.items():
        for layer_id, bank in enumerate(per_layer):
            bank.tensor.fill_(layer_id + 1)

    uploader = _FakeUploader({0, 2})
    os.environ["FREETOKEN_SKIP_BANK_PIN"] = "1"  # no CUDA in this test
    try:
        with resident_upload(uploader), PinPipeline() as pins:
            for layer_id in range(3):
                pins(layer_id, {n: per[layer_id] for n, per in banks.items()})
    finally:
        del os.environ["FREETOKEN_SKIP_BANK_PIN"]

    assert sorted(uploader.seen) == [0, 2]
    # uploaded contents are the pre-release bytes
    assert int(uploader.seen[0]["gate_up"][0, 0]) == 1
    assert int(uploader.seen[2]["down"][0, 0]) == 3
    # the unclaimed layer was never handed to the uploader
    assert 1 not in uploader.seen


def test_uploader_reports_layers_the_loader_skipped():
    from freetoken.moe.expert_banks import ResidentUploader

    up = ResidentUploader(frozenset({0, 1, 2}), torch.device("cpu"))
    banks = {"gate_up": HostBank((2, 16), torch.uint8), "down": HostBank((2, 16), torch.uint8)}
    up.upload(0, banks)
    assert up.missing() == [1, 2]
    up.upload(1, banks)
    up.upload(2, banks)
    assert up.missing() == []


# ---------------------------------------------------------------------------
# cache wiring
# ---------------------------------------------------------------------------


def _cache(num_layers=4, num_experts=8, cache_size=8, device="cpu"):
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=torch.device(device),
        quant_format="q4_0",
        prefill_overlap=False,
    )


def _sources(num_layers=4, num_experts=8, row=32):
    return {
        "gate_up": [torch.zeros(num_experts, row, dtype=torch.uint8) for _ in range(num_layers)],
        "down": [torch.zeros(num_experts, row, dtype=torch.uint8) for _ in range(num_layers)],
    }


def test_resident_layers_are_flagged_and_views_route_to_vram():
    cache = _cache()
    resident = {
        name: {l: torch.full((8, 32), l + 1, dtype=torch.uint8) for l in (0, 3)}
        for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({0, 3}))
    cache.set_bank_sources(_sources())

    assert cache.is_resident_layer(0) and cache.is_resident_layer(3)
    assert not cache.is_resident_layer(1)
    gate_up, down = cache.resident_views(3)
    assert gate_up.shape == (8, 32) and int(down[0, 0]) == 4
    # registration order matters: the kernels unpack views positionally
    assert cache.bank_schema == ("gate_up", "down")


def test_resident_layer_rejects_cpu_overlap():
    cache = _cache()
    cache.cpu_layer_ids = frozenset({1})
    resident = {name: {1: torch.zeros(8, 32, dtype=torch.uint8)} for name in ("gate_up", "down")}
    with pytest.raises(AssertionError, match="GPU-resident and CPU-decode"):
        cache.set_resident_banks(resident, frozenset({1}))


def test_resident_layer_requires_every_bank():
    cache = _cache()
    resident = {"gate_up": {0: torch.zeros(8, 32, dtype=torch.uint8)}}
    with pytest.raises(AssertionError, match="do not match"):
        cache.set_resident_banks(resident, frozenset({0}))


def test_set_bank_sources_tolerates_released_resident_sources():
    """The loader leaves resident layers' host banks in ``sources`` with their pages gone.

    They must survive validation (shape is still right) without being read.
    """
    cache = _cache()
    resident = {
        name: {0: torch.zeros(8, 32, dtype=torch.uint8)} for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({0}))
    sources = _sources()
    # a released HostBank keeps its shape but is not contiguous-checked by us any more;
    # simulate the worst case with a non-contiguous stand-in for the resident layer only
    sources["gate_up"][0] = torch.zeros(8, 64, dtype=torch.uint8)[:, ::2]
    cache.set_bank_sources(sources)  # must not raise
    assert cache.is_resident_layer(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the copy plan")
def test_copy_plan_nulls_resident_sources():
    """The fused copy descriptor must carry 0 for a resident layer's source pointer.

    Taking its device alias would either fail (the bank was never registered) or, worse,
    point at dropped pages.
    """
    device = torch.device("cuda")
    cache = _cache(device="cuda")
    resident = {
        name: {2: torch.zeros(8, 32, dtype=torch.uint8, device=device)}
        for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({2}))

    from freetoken.moe.host_banks import HostBank as HB

    banks = {name: [HB((8, 32), torch.uint8) for _ in range(4)] for name in ("gate_up", "down")}
    for per_layer in banks.values():
        for layer_id, b in enumerate(per_layer):
            if layer_id != 2:
                b.pin()
    cache.set_bank_sources({n: [b.tensor for b in per] for n, per in banks.items()})

    if cache._copy_fused_ok:
        assert cache._copy_src_ptrs_host[2] == [0, 0]
        assert all(p != 0 for p in cache._copy_src_ptrs_host[1])
