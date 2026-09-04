"""Device capability probing and JIT architecture selection.

Both exist for the same reason: a machine's accelerators need not all be servable by the
same build, and the failure mode when they are not is a fault rather than an exception. So
the arch flags have to follow the *devices present*, and anything that could fault has to
be found out about from a child process.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.device_probe import DeviceProbe, torch_has_code_for


def test_torch_has_code_for_matches_the_build_list():
    arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    if not arch_list:
        # No accelerator build: the veto must abstain, not refuse everything.
        assert torch_has_code_for("gfx90c")
        return
    assert torch_has_code_for(arch_list[0])
    assert not torch_has_code_for("gfx_definitely_not_a_real_arch")


def test_torch_has_code_for_ignores_target_id_features():
    """Devices report gfx1200:sramecc-:xnack-; build lists carry the bare name."""
    arch_list = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    if not arch_list:
        return
    base = arch_list[0].split(":", 1)[0]
    assert torch_has_code_for(f"{base}:xnack-:sramecc+")


def test_probe_renders_readably():
    assert "usable" in str(DeviceProbe(1, "gfx90c", True))
    bad = str(DeviceProbe(1, "gfx90c", False, "probe died on signal 11"))
    assert "unusable" in bad and "signal 11" in bad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_visible_device_archs_covers_every_device():
    from freetoken.kernel.gguf import device_arch, visible_device_archs

    archs = visible_device_archs()
    assert archs, "a visible device must contribute an arch"
    assert len(archs) == len(set(archs)), "archs must be deduplicated"
    for i in range(torch.cuda.device_count()):
        assert device_arch(i) in archs, f"device {i} has no compile target"
    # order is first-appearance, so the build directory hash stays stable across runs
    assert archs == visible_device_archs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_device_arch_drops_target_id_features():
    from freetoken.kernel.gguf import device_arch

    assert ":" not in device_arch(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_arch_selection_targets_present_devices_not_the_build_list(monkeypatch):
    """The whole point: the build follows the hardware, not what this wheel was built for.

    It has to go through PYTORCH_ROCM_ARCH. Passing --offload-arch in extra_cuda_cflags
    silently yields the *union* of torch's list and ours, because torch computes its flags
    before our extras are appended.
    """
    from freetoken.kernel.gguf import _apply_arch_selection, visible_device_archs

    is_hip = bool(getattr(torch.version, "hip", None))
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    archs = _apply_arch_selection(is_hip)
    if not is_hip:
        assert archs == [], "CUDA already derives its flags from the present devices"
        assert "PYTORCH_ROCM_ARCH" not in os.environ
        return
    assert archs == visible_device_archs()
    assert os.environ["PYTORCH_ROCM_ARCH"] == ";".join(archs)


def test_arch_selection_defers_to_an_explicit_choice(monkeypatch):
    """Cross-building for a device that is not in this box has to stay possible."""
    from freetoken.kernel.gguf import _apply_arch_selection

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx942")
    assert _apply_arch_selection(True) == []
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx942"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second device")
def test_probe_reports_a_second_device_without_killing_us():
    """A second device that cannot run a kernel must come back as a verdict, not a crash."""
    from freetoken.moe.device_probe import probe_device

    result = probe_device(1, timeout=180)
    assert result.index == 1 and result.arch
    if not result.usable:
        assert result.reason, "an unusable verdict has to say why"
