"""Host buffers two processes can both see, and both hand to their own GPU.

Some accelerators in a machine cannot be driven from the process that drives the others.
The reason is usually environmental rather than logical -- a runtime variable that selects
an architecture, a library path, a driver shim -- and such variables are read once per
process, so honouring one device's needs breaks the other's. The way out is to give that
device a worker process of its own; the way to make that cheap is to let both processes
address the same host memory instead of copying activations through a pipe.

Two properties make that work:

* **The mapping is shared, not private.** A ``MAP_SHARED`` file mapping is one set of
  physical pages with a mapping in each process. An anonymous mapping cannot be shared
  after a spawn, which is why these are file-backed even when nothing is ever read back
  from disk (the file lives in ``/dev/shm`` by default, so it never touches a disk).
* **Page-locking is per process, not per page.** Each side registers its *own* mapping with
  its *own* runtime, and each gets a device address for the same bytes. Neither has to know
  the other did it.

Nothing here is specific to a kind of device, a vendor, or a machine's topology: it is one
buffer, addressable by any number of processes, each of which may pin it for whatever
accelerator it drives.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import tempfile

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_PAGE = 4096

# Shared buffers default to a tmpfs so a "file" mapping costs RAM and never disk I/O.
# Overridable for hosts that mount /dev/shm too small for the expert traffic.
_SHM_DIR_ENV = "FREETOKEN_SHM_DIR"


def shm_dir() -> str:
    """Directory backing shared host buffers. Must be a tmpfs to stay off disk."""
    d = os.environ.get(_SHM_DIR_ENV, "").strip()
    if d:
        return d
    return "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()


class SharedHostBuffer:
    """A page-aligned host buffer backed by a shared file, mappable from several processes.

    The creating process makes the file and keeps its descriptor; a peer opens the same
    path. The path is only unlinked once every peer has mapped it -- unlinking earlier
    (the usual trick for scratch files) would leave a peer with no way to find it.

    ``pin()`` registers this process's mapping with the local accelerator runtime. It is
    deliberately separate from construction: a worker that only reads the buffer on the CPU
    never needs to spend pin quota on it.
    """

    __slots__ = ("path", "nbytes", "tensor", "addr", "_file", "_buf", "_pinned", "_owner")

    def __init__(self, path: str, shape: tuple[int, ...], dtype: torch.dtype, *, create: bool):
        elsize = torch.empty((), dtype=dtype).element_size()
        nbytes = 1
        for d in shape:
            nbytes *= d
        nbytes *= elsize
        self.nbytes = nbytes
        asize = ((nbytes + _PAGE - 1) // _PAGE) * _PAGE
        self.path = path
        self._owner = create

        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        self._file = os.fdopen(fd, "r+b")
        if create:
            os.ftruncate(fd, asize)
        self._buf = mmap.mmap(fd, asize)  # MAP_SHARED: the point of the exercise
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        assert self.addr % _PAGE == 0, "mmap must hand back a page-aligned address"
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=nbytes // elsize).view(*shape)
        self._pinned = False

    def pin(self) -> None:
        """Register this process's mapping so the local accelerator can address it."""
        if self._pinned:
            return
        from freetoken.kernel.pinned import host_register

        host_register(self.addr, len(self._buf))
        self._pinned = True

    def unlink(self) -> None:
        """Remove the name. Call only once every peer has mapped it; the mapping survives."""
        if not self._owner:
            return
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        # The torch view aliases the mapping, so it has to go before the mapping does.
        self.tensor = None
        try:
            self._buf.close()
        except BufferError:
            # Something still holds a memoryview of it; leaking the mapping until process
            # exit beats raising here, which would mask whatever the real failure was.
            logger.warning(f"shared buffer {self.path} still referenced at close")
        self._file.close()
        self.unlink()

    def __enter__(self) -> "SharedHostBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def create_shared(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> SharedHostBuffer:
    """Make a new shared buffer under :func:`shm_dir` (fails if the name is taken)."""
    return SharedHostBuffer(os.path.join(shm_dir(), name), shape, dtype, create=True)


def open_shared(path: str, shape: tuple[int, ...], dtype: torch.dtype) -> SharedHostBuffer:
    """Map a shared buffer another process created."""
    return SharedHostBuffer(path, shape, dtype, create=False)


__all__ = ["SharedHostBuffer", "create_shared", "open_shared", "shm_dir"]
