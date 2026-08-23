#pragma once

#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

namespace device {

inline constexpr auto kWarpThreads = 32u;

template <std::integral T, std::integral U>
__always_inline __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

namespace PDL {

template <bool kUsePDL> __always_inline __device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL> __always_inline __device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

} // namespace PDL

} // namespace device

namespace host {

inline auto
CUDA_CHECK(::hipError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::hipSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::hipGetErrorString(error));
  }
}

inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::hipGetLastError(), location);
}

template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    CUDA_CHECK(::hipFuncSetAttribute(
        F, ::hipFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}

struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device),
                               dynamic_shared_mem_bytes)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, hipStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream,
                               dynamic_shared_mem_bytes)) {}

  static auto resolve_device(DLDevice device) -> hipStream_t {
    return static_cast<hipStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CUDA_CHECK(
        ::hipLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

  auto with_attr([[maybe_unused]] bool use_pdl) -> LaunchKernel & {
#if defined(__HIP_PLATFORM_AMD__)
    // Programmatic Dependent Launch has no HIP counterpart: hipLaunchAttributeID
    // stops at MemSyncDomain and AMD hardware has no equivalent of the Hopper
    // programmatic-stream-serialization path. PDL only lets a dependent kernel
    // start its prologue early, so dropping it costs launch latency, never
    // correctness -- the stream ordering that guarantees the result is unchanged.
    m_config.numAttrs = 0;
#else
    if (use_pdl) {
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
#endif
    return *this;
  }

private:
  static auto s_make_config(dim3 grid_dim, dim3 block_dim, hipStream_t stream,
                            std::size_t smem) -> hipLaunchConfig_t {
    auto config = ::hipLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
  hipLaunchConfig_t m_config;
#if !defined(__HIP_PLATFORM_AMD__)
  cudaLaunchAttribute m_attr_cache;
#endif
};

} // namespace host
