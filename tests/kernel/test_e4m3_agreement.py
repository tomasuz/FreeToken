"""The host and the kernel must agree about native fp8.

They ask separately -- one before the launch to decide how to hand the tensor over, one
inside the kernel to decide which branch to compile -- and a disagreement is not a wrong
answer but a type error deep in a Triton trace, naming neither the device nor either
detector. That is exactly what a compute-capability check produces outside CUDA, where the
tensor library reports a fabricated capability for AMD parts.
"""

from freetoken.kernel.triton.e4m3_compat import e4m3_native, e4m3_native_cx


def test_the_host_and_the_kernel_agree_about_native_fp8():
    assert e4m3_native() == e4m3_native_cx()


def test_the_kernel_view_matches_the_branch_the_kernel_compiles():
    """A native branch takes fp8 straight; an emulated one needs the raw bytes."""
    import torch

    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view

    t = torch.zeros(4, dtype=torch.float8_e4m3fn)
    viewed = e4m3_kernel_view(t)

    if e4m3_native_cx():
        assert viewed.dtype == torch.float8_e4m3fn
    else:
        assert viewed.dtype == torch.uint8, (
            "the kernel compiles the emulation branch, which reads raw bytes"
        )
