"""Host buffers shared between processes.

The property that matters is the one a pipe would not give: a write in one process is
visible in another without anybody copying it. Everything else here guards the ways that
silently stops being true -- a private mapping instead of a shared one, an unlink that
happens before the peer has opened the path, a torch view outliving its mapping.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import torch

from freetoken.moe.shared_host import create_shared, open_shared, shm_dir


def _name() -> str:
    return f"freetoken-test-{uuid.uuid4().hex}"


def test_write_in_one_process_is_visible_in_another():
    buf = create_shared(_name(), (64,), torch.int32)
    try:
        buf.tensor.copy_(torch.arange(64, dtype=torch.int32))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src = (
            f"import torch,sys;sys.path.insert(0, {root!r});"
            "from freetoken.moe.shared_host import open_shared;"
            f"b = open_shared({buf.path!r}, (64,), torch.int32);"
            "print(int(b.tensor.sum()))"
        )
        out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-500:]
        assert int(out.stdout.strip()) == sum(range(64))
    finally:
        buf.close()


def test_peer_writes_are_seen_by_the_creator():
    """The direction that proves it is one set of pages, not a copy handed out at open."""
    buf = create_shared(_name(), (16,), torch.int32)
    try:
        buf.tensor.zero_()
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src = (
            f"import torch,sys;sys.path.insert(0, {root!r});"
            "from freetoken.moe.shared_host import open_shared;"
            f"b = open_shared({buf.path!r}, (16,), torch.int32);"
            "b.tensor.fill_(7)"
        )
        out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-500:]
        assert int(buf.tensor.sum()) == 7 * 16, "the creator must see the peer's write"
    finally:
        buf.close()


def test_buffer_is_page_aligned_and_right_sized():
    buf = create_shared(_name(), (3, 5), torch.float32)
    try:
        assert buf.addr % 4096 == 0
        assert buf.nbytes == 3 * 5 * 4
        assert buf.tensor.shape == (3, 5)
    finally:
        buf.close()


def test_name_collision_is_refused():
    """Silently adopting an existing buffer would hand two runs the same memory."""
    name = _name()
    buf = create_shared(name, (4,), torch.int32)
    try:
        try:
            create_shared(name, (4,), torch.int32)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a taken name must not be reused")
    finally:
        buf.close()


def test_close_unlinks_only_for_the_owner():
    name = _name()
    owner = create_shared(name, (4,), torch.int32)
    path = owner.path
    peer = open_shared(path, (4,), torch.int32)
    peer.close()  # a peer closing must not take the name away from the owner
    assert os.path.exists(path), "a peer's close must not unlink the buffer"
    owner.close()
    assert not os.path.exists(path), "the owner's close removes the name"


def test_shm_dir_prefers_tmpfs(monkeypatch):
    monkeypatch.delenv("FREETOKEN_SHM_DIR", raising=False)
    d = shm_dir()
    assert os.path.isdir(d)
    monkeypatch.setenv("FREETOKEN_SHM_DIR", "/tmp")
    assert shm_dir() == "/tmp"
