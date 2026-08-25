"""File-backed (spillable) host expert banks.

A spilled bank must be indistinguishable from the anonymous one for its consumers, and
``release()`` must give the bytes back to the filesystem -- not just drop the RAM copy and
leave the spill file fully allocated.
"""

from __future__ import annotations

import os

import torch

from freetoken.moe.host_banks import HostBank, bank_backing_default


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
# per-layer backing: spill only what can actually be reclaimed
# ---------------------------------------------------------------------------


def _specs(row=32):
    return {"gate_up": ((2, row), torch.uint8), "down": ((2, row), torch.uint8)}


def test_pinned_layers_are_not_spilled(tmp_path):
    """Registering a bank pins its pages, so a spill file would add writeback for nothing."""
    from freetoken.moe.host_banks import HostResidency, alloc_layer_banks, requested_residency

    os.environ["FREETOKEN_BANK_SPILL_DIR"] = str(tmp_path)
    try:
        labels = [HostResidency.PINNED.value, HostResidency.PAGEABLE.value,
                  HostResidency.LOCKED.value]
        with requested_residency(labels):
            banks = alloc_layer_banks(_specs(), 3)
        spilled = [b._spill is not None for b in banks["gate_up"]]
        assert spilled == [False, True, False], spilled
    finally:
        del os.environ["FREETOKEN_BANK_SPILL_DIR"]


def test_resident_layers_are_not_spilled(tmp_path):
    """A resident layer's bank is staging: filled, uploaded, released. Never re-read.

    Layer 1 is marked pageable so the resident veto is what the assertion isolates -- with
    no residency plan every layer pins, and pinning already vetoes spilling on its own.
    """
    from freetoken.moe.host_banks import (
        HostResidency,
        alloc_layer_banks,
        requested_residency,
        resident_upload,
    )

    class _Up:
        layers = frozenset({0, 2})

        def claims(self, layer_id):
            return layer_id in self.layers

    os.environ["FREETOKEN_BANK_SPILL_DIR"] = str(tmp_path)
    try:
        labels = [HostResidency.PAGEABLE.value] * 3  # all three would spill on their own
        with requested_residency(labels), resident_upload(_Up()):
            banks = alloc_layer_banks(_specs(), 3)
        spilled = [b._spill is not None for b in banks["gate_up"]]
        assert spilled == [False, True, False], spilled
    finally:
        del os.environ["FREETOKEN_BANK_SPILL_DIR"]


def test_no_spill_dir_means_no_spill_files():
    from freetoken.moe.host_banks import alloc_layer_banks

    banks = alloc_layer_banks(_specs(), 3)
    assert all(b._spill is None for b in banks["gate_up"])
