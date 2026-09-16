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
#include <cstdint>
#include <sched.h>

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


// --- the same handshake, from the host ------------------------------------------------
//
// A layer whose routes all belong to someone else should cost this device nothing. Whether
// that is so is already written in host memory -- the engine fills the routing before it
// rings -- so the decision belongs to the host, before a single kernel is launched. These
// three do from the CPU what the kernels above do from the device.
//
// The wait releases the GIL. It sits here for as long as a layer takes, and the engine's
// own Python has to keep running while it does.

bool handshake_host_wait(int64_t flag_addr, int64_t slot, int64_t max_spins) {
  u64 *p = reinterpret_cast<u64 *>(flag_addr) + slot;
  pybind11::gil_scoped_release release;
  for (int64_t n = 0; n < max_spins; ++n) {
    if (__atomic_load_n(p, __ATOMIC_ACQUIRE) >= 1ULL) {
      return true;
    }
    if ((n & 0x3F) == 0x3F) {
      sched_yield();
    } else {
#if defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#endif
    }
  }
  return false;
}

// Clear one word and raise another, in that order. The release on the second is what makes
// everything written before it -- the zeroed answer of a layer with no routes -- visible to
// whoever is waiting on it.
void handshake_host_raise(int64_t clear_addr, int64_t raise_addr, int64_t slot) {
  __atomic_store_n(reinterpret_cast<u64 *>(clear_addr) + slot, 0ULL, __ATOMIC_RELAXED);
  __atomic_store_n(reinterpret_cast<u64 *>(raise_addr) + slot, 1ULL, __ATOMIC_RELEASE);
}

// How many routes of this step are this executor's: the ids are int32 and a route that
// belongs to someone else is -1.
int64_t handshake_count_nonneg_i32(int64_t addr, int64_t n) {
  const int32_t *p = reinterpret_cast<const int32_t *>(addr);
  int64_t k = 0;
  for (int64_t i = 0; i < n; ++i) {
    k += (p[i] >= 0) ? 1 : 0;
  }
  return k;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("doorbell", &handshake_doorbell, "ring the worker's doorbell for one slot");
  m.def("wait", &handshake_wait, "hold the stream until the worker reports that slot done");
  m.def("host_wait", &handshake_host_wait, "spin on the host until this slot's flag is raised");
  m.def("host_raise", &handshake_host_raise, "clear one flag and raise another, from the host");
  m.def("count_nonneg_i32", &handshake_count_nonneg_i32, "how many int32 entries are >= 0");
}
