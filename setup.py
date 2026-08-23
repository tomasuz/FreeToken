from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension

try:
    from torch.utils.cpp_extension import ROCM_HOME
except ImportError:  # older torch
    ROCM_HOME = None

# torch's ROCm builds report HIP through torch.version.hip; CUDA_HOME is unset there.
import torch

IS_ROCM = bool(getattr(torch.version, "hip", None)) and ROCM_HOME is not None


ROOT = Path(__file__).parent


def _check_toolchain() -> None:
    if IS_ROCM:
        # hipcc ships with the ROCm that torch was built against; there is no
        # nvcc/torch version pairing to verify.
        return
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if IS_ROCM:
        rocm_home = Path(ROCM_HOME)
        library_dirs = [str(rocm_home / "lib")]
        lib64 = rocm_home / "lib64"
        if lib64.exists():
            library_dirs.append(str(lib64))
        # On Debian ROCM_HOME is /usr, and handing clang an explicit
        # "-isystem /usr/include" pulls that directory out of its default
        # ordering, after which libstdc++'s <cmath> can no longer resolve its
        # #include_next <math.h>. The HIP headers live on the default path
        # there anyway, so contribute no include dir at all in that case.
        include_dirs = []
        if rocm_home != Path("/usr"):
            include_dirs.append(str(rocm_home / "include"))
        return include_dirs, library_dirs
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


_RUNTIME_LIBS = ["amdhip64"] if IS_ROCM else ["cudart"]
cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=_RUNTIME_LIBS,
            extra_compile_args=["-O3", "-std=c++17"],
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=_RUNTIME_LIBS,
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
