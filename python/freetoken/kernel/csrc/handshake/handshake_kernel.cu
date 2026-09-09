// The worker doorbell and wait, as kernels rather than stream memory operations.
//
// The handshake was written with hipStreamWriteValue64/hipStreamWaitValue64, which is the
// natural expression of it and works correctly outside a graph. Inside a capture this
// runtime accepts both calls and records NOTHING -- torch reports "The CUDA Graph is
// empty" and a replay performs neither. That is why a captured decode with a worker
// produced garbage: the doorbell was never rung, so the worker never ran, and the wait
// never blocked, so the engine read whatever the output buffer happened to hold. With the
// worker's share at zero the garbage was multiplied by zero, which is why the fault
// looked like it lived in the worker's contribution rather than in the handshake.
//
// A kernel is an ordinary graph node. These two do exactly what the memory operations did,
// and a capture records them like anything else.
//
// The flags live in host memory the driver has page-locked, so every access here is
// system-scope: the other side of this handshake is a CPU in a different process, and a
// device-scope atomic would let the GPU keep its own view of a word only the host writes.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

using u64 = unsigned long long;

__device__ __forceinline__ u64 load_system(const u64 *p) {
#ifdef __HIP_PLATFORM_AMD__
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  u64 v = *reinterpret_cast<const volatile u64 *>(p);
  __threadfence_system();
  return v;
#endif
}

__device__ __forceinline__ void store_system(u64 *p, u64 v) {
#ifdef __HIP_PLATFORM_AMD__
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  __threadfence_system();
  *reinterpret_cast<volatile u64 *>(p) = v;
#endif
}

__device__ __forceinline__ void pause_briefly() {
#ifdef __HIP_PLATFORM_AMD__
  __builtin_amdgcn_s_sleep(8);
#elif __CUDA_ARCH__ >= 700
  __nanosleep(200);
#endif
}

// done[slot] = 0, then ready[slot] = 1. The order is the whole protocol: clearing done
// after raising ready would wipe the completion the worker is about to write for THIS
// step, and the release on ready is what makes the cleared done -- and the activation
// copies this kernel is stream-ordered behind -- visible to the worker before it starts.
__global__ void doorbell_kernel(u64 *done, u64 *ready, int slot) {
  if (threadIdx.x || blockIdx.x) return;
  store_system(done + slot, 0ULL);
  store_system(ready + slot, 1ULL);
}

// Hold the stream until the worker reports done. ``max_spins`` is a wedge guard, not a
// timeout with meaningful units: a worker that has died must not leave a graph replay
// spinning forever with no way to interrupt it. Giving up leaves the output buffer
// holding the previous step's bytes, which the host notices as a stalled slot.
__global__ void wait_kernel(const u64 *done, int slot, u64 max_spins) {
  if (threadIdx.x || blockIdx.x) return;
  for (u64 n = 0; n < max_spins; ++n) {
    if (load_system(done + slot) >= 1ULL) return;
    pause_briefly();
  }
}

}  // namespace

void handshake_doorbell(int64_t done_addr, int64_t ready_addr, int64_t slot) {
  auto stream = at::cuda::getCurrentCUDAStream();
  doorbell_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<u64 *>(done_addr),
                                       reinterpret_cast<u64 *>(ready_addr),
                                       static_cast<int>(slot));
}

void handshake_wait(int64_t done_addr, int64_t slot, int64_t max_spins) {
  auto stream = at::cuda::getCurrentCUDAStream();
  wait_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<const u64 *>(done_addr),
                                   static_cast<int>(slot), static_cast<u64>(max_spins));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("doorbell", &handshake_doorbell, "ring the worker's doorbell for one slot");
  m.def("wait", &handshake_wait, "hold the stream until the worker reports that slot done");
}
