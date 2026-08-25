"""Which accelerators may actually be handed expert work, on any machine.

Placing experts on a device is only safe if two independent things hold, and neither is
implied by the device merely being visible:

* **This PyTorch has code for it.** A build's architecture list is fixed at build time and
  need not cover every device in the box. A missing architecture is not a clean error --
  the first kernel launch faults.
* **The device actually runs a kernel.** Driver quirks, architectures that report as
  supported but are not, and libraries missing a per-architecture data file all show up
  only when something runs.

So the second check runs in a **subprocess**. A fault or a hang in the child is data; the
same fault in-process is the end of the server, and no ``try``/``except`` changes that.

Nothing here knows what kind of device it is looking at. A machine may have one
accelerator or several, of one kind or several; the probe answers per device and the
placement planner decides what to do with the answer.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Long enough for a cold driver + context init on a slow device, short enough that probing
# a boxful of them does not dominate startup. A device that cannot allocate and run one
# trivial kernel inside this is not a device we want serving experts anyway.
_PROBE_TIMEOUT_S = 120

# Runs in the child, with only the device under test visible (hence index 0 there).
_PROBE_SRC = """
import torch
t = torch.zeros(256, 256, dtype=torch.float16, device="cuda")
t.add_(1)
torch.cuda.synchronize()
assert float(t.sum()) > 0
print("ok")
"""


@dataclass(frozen=True)
class DeviceProbe:
    index: int
    arch: str
    usable: bool
    reason: str = ""

    def __str__(self) -> str:
        state = "usable" if self.usable else f"unusable ({self.reason})"
        return f"device {self.index} [{self.arch}]: {state}"


def _visibility_env_var() -> str:
    """The env var that restricts device visibility for this build's runtime."""
    import torch

    return "HIP_VISIBLE_DEVICES" if getattr(torch.version, "hip", None) else "CUDA_VISIBLE_DEVICES"


def torch_has_code_for(arch: str) -> bool:
    """Whether this PyTorch build carries device code for ``arch``.

    A cheap, in-process veto that catches the common case before paying for a subprocess.
    It is a veto only: passing does not prove the device works, which is what the probe is
    for. An empty/unknown arch list means "cannot tell" -- answer yes and let the probe
    decide, rather than refusing a device on missing metadata.
    """
    import torch

    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:  # no accelerator build, or an API that does not exist here
        return True
    if not arch_list:
        return True
    # ROCm entries are plain gfx names, CUDA entries are sm_XX / compute_XX; both compare
    # after dropping any target-ID features.
    supported = {a.split(":", 1)[0] for a in arch_list}
    return arch.split(":", 1)[0] in supported


def probe_device(index: int, timeout: float = _PROBE_TIMEOUT_S) -> DeviceProbe:
    """Allocate and run one trivial kernel on ``index``, in a child process."""
    from freetoken.kernel.gguf import device_arch

    arch = device_arch(index)
    if not torch_has_code_for(arch):
        return DeviceProbe(
            index, arch, False,
            f"this PyTorch build has no device code for {arch} "
            f"(built for: {', '.join(_arch_list())})",
        )

    env = dict(os.environ)
    env[_visibility_env_var()] = str(index)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SRC],
            env=env, timeout=timeout, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return DeviceProbe(index, arch, False, f"probe hung for {timeout:.0f}s")

    if proc.returncode == 0:
        return DeviceProbe(index, arch, True)
    if proc.returncode < 0:
        return DeviceProbe(index, arch, False, f"probe died on signal {-proc.returncode}")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1] if tail else f"exit {proc.returncode}"
    return DeviceProbe(index, arch, False, detail[:200])


def _arch_list() -> list[str]:
    import torch

    try:
        return list(torch.cuda.get_arch_list())
    except Exception:
        return []


def probe_all_devices(timeout: float = _PROBE_TIMEOUT_S) -> list[DeviceProbe]:
    """Probe every visible device. Index 0 is assumed usable -- the engine is already
    running on it, so a probe could only confirm what the process has proven by existing,
    and paying for a subprocess to learn that is waste."""
    import torch

    if not torch.cuda.is_available():
        return []
    from freetoken.kernel.gguf import device_arch

    results = [DeviceProbe(0, device_arch(0), True)]
    results += [probe_device(i, timeout) for i in range(1, torch.cuda.device_count())]
    for r in results:
        (logger.info_rank0 if r.usable else logger.warning_rank0)(f"expert device probe: {r}")
    return results


def usable_expert_devices(timeout: float = _PROBE_TIMEOUT_S) -> list[int]:
    """Indices of devices that may be given expert layers."""
    return [p.index for p in probe_all_devices(timeout) if p.usable]


__all__ = [
    "DeviceProbe",
    "probe_device",
    "probe_all_devices",
    "torch_has_code_for",
    "usable_expert_devices",
]
