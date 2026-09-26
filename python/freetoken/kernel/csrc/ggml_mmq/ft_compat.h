// Force-included (-include) ahead of the vendored llama.cpp sources.
//
// The MMQ templates call integer min()/max() in device code. With the system HIP headers
// those come from clang's __clang_hip_math.h (via <hip/hip_runtime.h>); the ROCm headers a
// ROCM_HOME can point at do not include it, and the templates then fail with "use of
// undeclared identifier 'min'". Supply the same device overloads clang would, only when
// clang's header has not.
#pragma once
// vendors/hip.h sets this before its own <hip/hip_runtime.h>: llama.cpp brings its own
// __shfl_*_sync, and HIP's (64-bit mask semantics) must stay out. Including the runtime
// here first without it would switch them back on under llama.cpp's warp code.
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__) && !defined(__CLANG_HIP_MATH_H__) && !defined(FREETOKEN_HIP_INT_MINMAX)
#define FREETOKEN_HIP_INT_MINMAX
template <class T> __device__ __forceinline__ T min(T a, T b) { return a < b ? a : b; }
template <class T> __device__ __forceinline__ T max(T a, T b) { return a > b ? a : b; }
__device__ __forceinline__ int min(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }
#endif
