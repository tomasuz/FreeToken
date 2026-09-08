// Grouped expert GEMV over native NVFP4 (ModelOpt) banks, for devices whose Triton
// backend cannot serve the production kernel.
//
// The production NVFP4 decode path is Triton (kernel/triton/nvfp4_fused_moe.py). Triton's
// AMD backend covers CDNA and RDNA3+; a GCN5-class part (gfx900/gfx90c -- an integrated
// GPU, say) is refused outright at compile time, which takes the whole device out of the
// MoE split rather than one kernel. hipcc has no such gap, so this file exists to give
// those devices the same arithmetic through a compiler that will build for them. It is a
// deliberate second implementation of one small kernel, and the Triton version stays the
// reference: tests/moe/test_nvfp4_moe_vec.py pins the two together.
//
// Weight layout, per the "nvfp4" bank schema:
//   packed [S, N, K/2]  uint8   two e2m1 codes per byte, low nibble first (even k)
//   scale  [S, N, K/16] uint8   one fp8-e4m3 scale per 16 consecutive k
//   global [S, N]       fp16    one per output row; folding it into the block scales
//                               would underflow, which is why it is a separate bank
// and the value of weight element (n, k) is  E2M1[code] * e4m3(scale) * global[n].
//
// The routed weight is NOT applied here: the caller multiplies it in, matching
// fused_experts_gguf so the two expert paths compose the same way.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

// E2M1 has 16 values and no arithmetic worth doing: a lookup beats decoding the bits.
__device__ __constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// fp8-e4m3 (bias 7, 3 mantissa bits) straight to float, subnormals included. Decoded
// arithmetically rather than through a half bitcast: this has to compile for targets with
// no half arithmetic at all, and the exponent arithmetic is exact on any of them.
__device__ __forceinline__ float e4m3_to_float(unsigned char v) {
  const int mant = v & 0x7;
  const int exp = (v >> 3) & 0xF;
  const float mag = (exp == 0)
      ? static_cast<float>(mant) * 0.001953125f          // 2^-9, the subnormal step
      : ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  return (v & 0x80) ? -mag : mag;
}

// Activations reach the kernel in the model's compute dtype. Both conversions are bit
// manipulation so that no half/bfloat16 instruction is required of the device.
__device__ __forceinline__ float to_float(const at::BFloat16 &x) {
  unsigned int bits = static_cast<unsigned int>(*reinterpret_cast<const unsigned short *>(&x)) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}
__device__ __forceinline__ float to_float(const at::Half &x) { return __half2float(*reinterpret_cast<const __half *>(&x)); }
__device__ __forceinline__ float to_float(const float &x) { return x; }

__device__ __forceinline__ void from_float(float v, at::BFloat16 *out) {
  unsigned int bits;
  memcpy(&bits, &v, sizeof(bits));
  // round-to-nearest-even on the truncated 16 bits
  const unsigned int rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  const unsigned short hi = static_cast<unsigned short>(rounded >> 16);
  memcpy(out, &hi, sizeof(hi));
}
__device__ __forceinline__ void from_float(float v, at::Half *out) {
  const __half h = __float2half(v);
  memcpy(out, &h, sizeof(h));
}
__device__ __forceinline__ void from_float(float v, float *out) { *out = v; }

// One wave per output row: lanes stride the K axis by whole bytes, so consecutive lanes
// read consecutive bytes of the same weight row and the packed bank is read coalesced.
// A block carries several waves to keep occupancy up without splitting a reduction.
template <typename scalar_t>
__global__ void nvfp4_moe_vec_kernel(
    const scalar_t *__restrict__ a,        // [rows_a, K]
    const unsigned char *__restrict__ packed,  // [S, N, K/2]
    const unsigned char *__restrict__ scale,   // [S, N, K/16]
    const at::Half *__restrict__ global_,      // [S, N]
    const int *__restrict__ topk_ids,          // [routes]
    scalar_t *__restrict__ out,                // [routes, N]
    const int routes, const int N, const int K, const int top_k) {
  const int wave = warpSize;
  const int lane = threadIdx.x % wave;
  const int wave_in_block = threadIdx.x / wave;
  const int waves_per_block = blockDim.x / wave;

  const int route = blockIdx.x;
  const int n = blockIdx.y * waves_per_block + wave_in_block;
  if (route >= routes || n >= N) return;

  const int slot = topk_ids[route];
  if (slot < 0) return;  // a route another executor owns; its weight is zero anyway
  // top_k == 1 means the caller already expanded one activation row per route.
  const int a_row = route / top_k;

  const int k_bytes = K / 2;
  const long w_row = (static_cast<long>(slot) * N + n);
  const unsigned char *p_row = packed + w_row * k_bytes;
  const unsigned char *s_row = scale + w_row * (K / 16);
  const scalar_t *a_row_p = a + static_cast<long>(a_row) * K;

  float acc = 0.0f;
  for (int kb = lane; kb < k_bytes; kb += wave) {
    const unsigned char byte = p_row[kb];
    const float s = e4m3_to_float(s_row[kb >> 3]);  // 8 bytes == 16 k == one scale
    acc += to_float(a_row_p[2 * kb]) * kE2M1[byte & 0xF] * s;
    acc += to_float(a_row_p[2 * kb + 1]) * kE2M1[(byte >> 4) & 0xF] * s;
  }
  // HIP's __shfl_down carries no mask; CUDA's unmasked form is removed, so each side
  // gets the spelling it still accepts. Every lane of the wave reaches this loop.
  for (int off = wave / 2; off > 0; off >>= 1) {
#ifdef __HIP_PLATFORM_AMD__
    acc += __shfl_down(acc, off, wave);
#else
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, off, wave);
#endif
  }

  if (lane == 0) {
    acc *= __half2float(*reinterpret_cast<const __half *>(&global_[w_row]));
    from_float(acc, &out[static_cast<long>(route) * N + n]);
  }
}

}  // namespace

torch::Tensor nvfp4_moe_vec(torch::Tensor a, torch::Tensor packed, torch::Tensor scale,
                            torch::Tensor global_, torch::Tensor topk_ids, int64_t top_k,
                            int64_t row, int64_t tokens) {
  TORCH_CHECK(a.is_cuda() && packed.is_cuda() && scale.is_cuda() && global_.is_cuda());
  TORCH_CHECK(a.dim() == 2, "activations must be [rows, K]");
  TORCH_CHECK(packed.dim() == 3 && scale.dim() == 3 && global_.dim() == 2);
  TORCH_CHECK(packed.scalar_type() == at::kByte && scale.scalar_type() == at::kByte,
              "packed codes and e4m3 block scales are read as raw bytes");
  TORCH_CHECK(global_.scalar_type() == at::kHalf, "per-row globals are fp16");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt, "expert/slot ids must be int32");
  const int K = static_cast<int>(a.size(1));
  TORCH_CHECK(K % 16 == 0, "K must be a whole number of 16-wide scale blocks, got ", K);
  TORCH_CHECK(packed.size(2) == K / 2 && scale.size(2) == K / 16,
              "weight banks disagree with the activation's K");
  TORCH_CHECK(packed.size(1) == row && global_.size(1) == row);

  const at::cuda::CUDAGuard guard(a.device());
  const int64_t routes = tokens * top_k;
  auto out = torch::empty({routes, row}, a.options());
  if (routes == 0) return out;

  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(a.scalar_type(), "nvfp4_moe_vec",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        const int waves = threads / at::cuda::warp_size();
        dim3 grid(routes, (row + waves - 1) / waves);
        nvfp4_moe_vec_kernel<at::BFloat16><<<grid, threads, 0, stream>>>(
            a.data_ptr<at::BFloat16>(), packed.data_ptr<unsigned char>(),
            scale.data_ptr<unsigned char>(), global_.data_ptr<at::Half>(),
            topk_ids.data_ptr<int>(), out.data_ptr<at::BFloat16>(),
            static_cast<int>(routes), static_cast<int>(row), K, static_cast<int>(top_k));
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        const int waves = threads / at::cuda::warp_size();
        dim3 grid(routes, (row + waves - 1) / waves);
        nvfp4_moe_vec_kernel<at::Half><<<grid, threads, 0, stream>>>(
            a.data_ptr<at::Half>(), packed.data_ptr<unsigned char>(),
            scale.data_ptr<unsigned char>(), global_.data_ptr<at::Half>(),
            topk_ids.data_ptr<int>(), out.data_ptr<at::Half>(),
            static_cast<int>(routes), static_cast<int>(row), K, static_cast<int>(top_k));
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        const int waves = threads / at::cuda::warp_size();
        dim3 grid(routes, (row + waves - 1) / waves);
        nvfp4_moe_vec_kernel<float><<<grid, threads, 0, stream>>>(
            a.data_ptr<float>(), packed.data_ptr<unsigned char>(),
            scale.data_ptr<unsigned char>(), global_.data_ptr<at::Half>(),
            topk_ids.data_ptr<int>(), out.data_ptr<float>(),
            static_cast<int>(routes), static_cast<int>(row), K, static_cast<int>(top_k));
      }));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_moe_vec", &nvfp4_moe_vec,
        "Grouped expert GEMV over NVFP4 banks (e2m1 codes, e4m3 per-16 scales, fp16 row globals)");
}
