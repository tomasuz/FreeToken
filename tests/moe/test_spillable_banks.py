"""File-backed (spillable) host expert banks.

A spilled bank must be indistinguishable from the anonymous one for its consumers, and
``release()`` must give the bytes back to the filesystem -- not just drop the RAM copy and
leave the spill file fully allocated.
"""

from __future__ import annotations

import os

import pytest
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


# ---------------------------------------------------------------------------
# release() has to actually return the memory
# ---------------------------------------------------------------------------


def _meminfo(field: str) -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith(field):
                return int(line.split()[1]) / 1024  # MiB
    return 0.0


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="needs /proc/meminfo")
def test_release_actually_returns_the_pages():
    """The failure this guards was silent: release() ran, logged, and freed nothing.

    mmap(-1, n) maps MAP_SHARED, which is tmpfs-backed -- those pages count as Shmem and
    MADV_DONTNEED does not reclaim them. A bank "handed away" then stays resident for the
    life of the process, which on a tight host is the difference between a second process
    starting and being killed by the allocator.
    """
    size_mib = 512
    before = _meminfo("MemAvailable:")
    bank = HostBank((size_mib, 1024, 1024), torch.uint8)
    bank.tensor[:, ::4096, 0] = 1  # touch a page in every 4 MiB to fault them in
    bank.tensor.fill_(7)
    filled = _meminfo("MemAvailable:")
    assert before - filled > size_mib * 0.5, (
        f"filling {size_mib} MiB should show up in MemAvailable "
        f"(saw {before - filled:.0f} MiB)"
    )

    bank.release()
    after = _meminfo("MemAvailable:")
    recovered = after - filled
    assert recovered > size_mib * 0.5, (
        f"release() returned only {recovered:.0f} MiB of {size_mib} MiB -- the mapping is "
        f"probably shared, where MADV_DONTNEED does not reclaim"
    )


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="needs /proc/meminfo")
def test_anonymous_banks_do_not_land_in_shmem():
    """Shmem accounting is the tell: a shared mapping shows there, a private one does not."""
    before = _meminfo("Shmem:")
    bank = HostBank((256, 1024, 1024), torch.uint8)
    bank.tensor.fill_(3)
    grew = _meminfo("Shmem:") - before
    bank.release()
    assert grew < 64, f"anonymous bank added {grew:.0f} MiB of Shmem; it should add none"


# ---------------------------------------------------------------------------
# banks another process can map
# ---------------------------------------------------------------------------


def test_a_shared_bank_is_a_file_a_peer_can_open_by_name():
    """A worker on another device can only serve a layer it can reach the weights for."""
    import os

    from freetoken.moe.host_banks import HostBank

    bank = HostBank((4, 16), torch.uint8, backing="shared", shared_name="freetoken-test-open")
    bank.tensor.fill_(9)

    assert bank.shared_path and os.path.exists(bank.shared_path)
    with open(bank.shared_path, "rb") as fh:
        assert fh.read(4) == b"\x09\x09\x09\x09"

    os.unlink(bank.shared_path)


def test_two_mappings_of_a_shared_bank_see_the_same_bytes():
    """The point is one copy of the weights, not one copy per consumer."""
    import mmap
    import os

    from freetoken.moe.host_banks import HostBank

    bank = HostBank((4, 16), torch.uint8, backing="shared", shared_name="freetoken-test-share")
    bank.tensor.fill_(1)

    with open(bank.shared_path, "r+b") as fh:
        peer = mmap.mmap(fh.fileno(), 0)
        assert peer[0] == 1
        bank.tensor[0, 0] = 7  # a write through one mapping
        assert peer[0] == 7  # is visible through the other
        peer.close()

    os.unlink(bank.shared_path)


def test_a_shared_bank_still_releases_its_pages():
    """The reason banks were made private in the first place must not come back.

    An anonymous shared mapping counts as Shmem and ignores MADV_DONTNEED, so release()
    freed nothing. A named file has the answer that one lacked: punching a hole returns the
    blocks. Shareable and releasable are not a trade here.
    """
    import ctypes
    import os

    from freetoken.moe.host_banks import HostBank

    bank = HostBank((256, 4096), torch.uint8, backing="shared", shared_name="freetoken-test-rel")
    bank.tensor.fill_(3)

    libc = ctypes.CDLL(None, use_errno=True)
    pages = (len(bank._buf) + 4095) // 4096
    vec = (ctypes.c_ubyte * pages)()
    libc.mincore(ctypes.c_void_p(bank.addr), ctypes.c_size_t(len(bank._buf)), vec)
    assert sum(v & 1 for v in vec) > 0, "the fill should have made it resident"

    bank.release()

    vec = (ctypes.c_ubyte * pages)()
    libc.mincore(ctypes.c_void_p(bank.addr), ctypes.c_size_t(len(bank._buf)), vec)
    assert sum(v & 1 for v in vec) == 0, "release must return a shared bank's pages too"

    os.unlink(bank.shared_path)


def test_shared_banks_are_opt_in():
    """A single-process run should not be leaving files in a tmpfs for nobody."""
    from freetoken.moe.host_banks import _layer_backing, shared_banks

    assert _layer_backing(0) in (None, "mmap")
    with shared_banks("tag"):
        assert _layer_backing(0) == "shared"
    assert _layer_backing(0) in (None, "mmap")


def test_shared_banks_refuse_a_directory_too_small_to_hold_them(monkeypatch, tmp_path):
    """A sparse file in a full tmpfs fails on the write, as SIGBUS, with no message.

    That is how this first failed: the backend died during load with no traceback and no
    clue. The size is knowable before a byte is written, so it has to be checked there.
    """
    from freetoken.moe import host_banks

    monkeypatch.setattr("freetoken.moe.shared_host.shm_dir", lambda: str(tmp_path))
    fake = os.statvfs_result((4096, 4096, 100, 10, 10, 0, 0, 0, 0, 255))  # ~40 KiB free
    monkeypatch.setattr(host_banks.os, "statvfs", lambda _d: fake)

    specs = {"gate_up": ((64, 4096), torch.uint8)}
    with host_banks.shared_banks("tag"):
        with pytest.raises(RuntimeError, match="shared expert banks need"):
            host_banks.alloc_layer_banks(specs, 8)


def test_the_shortfall_message_names_ways_out_that_exist(monkeypatch, tmp_path):
    from freetoken.moe import host_banks

    monkeypatch.setattr("freetoken.moe.shared_host.shm_dir", lambda: str(tmp_path))
    fake = os.statvfs_result((4096, 4096, 100, 10, 10, 0, 0, 0, 0, 255))
    monkeypatch.setattr(host_banks.os, "statvfs", lambda _d: fake)

    with host_banks.shared_banks("tag"):
        with pytest.raises(RuntimeError) as caught:
            host_banks.alloc_layer_banks({"gate_up": ((64, 4096), torch.uint8)}, 8)

    text = str(caught.value)
    for way_out in ("FREETOKEN_SHM_DIR", "--moe-resident-layers", "--moe-worker-layers"):
        assert way_out in text
