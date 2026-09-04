#include "hip/hip_runtime.h"
// Minimal AT_DISPATCH helper for the vendored GGUF kernels (borrowed from
// sgl-kernel csrc/quantization/gguf, which are ports of llama.cpp). The donor
// pulls these macros from its large include/utils.h; we only need the float
// dispatch, so vendor just that to keep the JIT compile self-contained.
#pragma once

#include <ATen/Dispatch.h>

// Hardware wavefront width: how many lanes a cross-lane operation spans, and therefore how
// wide a block the MMVQ kernels have to be launched with.
//
// Not WARP_SIZE_GGUF, which is the quant block width -- 32 quants per GGUF block -- and
// describes the data, not the machine. The two coincide on the architectures these kernels
// came from, which is exactly why the difference is easy to miss. The MMQ kernels tile by
// the block width and never shuffle, so they stay 32 wide on every device; only the MMVQ
// ones care what the hardware does.
//
// Device code can hold it as a constant, because the device pass is compiled once per
// target. Host code cannot: one host object serves every target in a fat binary, so the
// launch sites ask the device instead (gguf_device_warp_size).
#ifndef WARP_SIZE
#  if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
#    define WARP_SIZE 64  // GCN and CDNA
#  else
#    define WARP_SIZE 32  // RDNA, and NVIDIA
#  endif
#endif

// Wavefront width of the device a launch will land on. Cached per device: a property query
// on every launch would show up on the decode path. Concurrent first callers race, and
// store the same value, so the race is benign.
static inline int gguf_device_warp_size() {
  constexpr int kMaxDevices = 16;
  static int cache[kMaxDevices] = {0};
  int dev = 0;
#if defined(__HIP_PLATFORM_AMD__)
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDevices) return WARP_SIZE;
  if (cache[dev] == 0) {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, dev) != hipSuccess) return WARP_SIZE;
    cache[dev] = props.warpSize;
  }
#else
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return WARP_SIZE;
  if (cache[dev] == 0) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, dev) != cudaSuccess) return WARP_SIZE;
    cache[dev] = props.warpSize;
  }
#endif
  return cache[dev];
}

// Warp-shuffle wrappers the donor pulls from sgl-kernel's utils.h (CUDA variants).

// "All lanes" differs by wavefront width: 32 bits on RDNA, 64 on GCN/CDNA.
#if defined(__HIP_PLATFORM_AMD__)
#define SGLANG_FULL_MASK (~0ull)
#else
#define SGLANG_FULL_MASK (uint32_t(-1))
#endif

// HIP's __shfl_xor_sync static_asserts on a 64-bit mask (wave64 has 64 lanes),
// so widen whatever the CUDA-shaped call sites pass.
#if defined(__HIP_PLATFORM_AMD__)
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) \
  __shfl_xor_sync(static_cast<unsigned long long>(mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync(static_cast<unsigned long long>(mask), (var), (lane_mask), (width))
#endif
#else
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))
#endif
#endif

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))
