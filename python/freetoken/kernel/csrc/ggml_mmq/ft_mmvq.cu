// FreeToken entry point for the vendored llama.cpp MMVQ (see UPSTREAM.md).
//
// mmvq.cu is included whole so its internal launcher (mul_mat_vec_q_switch_type) is callable;
// ft_mmvq() then does what ggml_cuda_mul_mat_vec_q() does -- quantize the activations to
// q8_1 and launch -- from caller pointers and strides instead of ggml_tensors. The runtime
// symbols the headers need come from ft_mmq.cu; the few ggml.c helpers that
// ggml_cuda_mul_mat_vec_q() itself references (compiled in, never called here) are stubbed
// below so the library links.

#include "ggml-cuda/mmvq.cu"

static size_t ft_mmvq_type_size(ggml_type type, int * block);

extern "C" {
// used on live paths (quantize / launch sizing): real answers for the served types
int64_t ggml_blck_size(enum ggml_type type) {
    int block = 1;
    return ft_mmvq_type_size(type, &block) ? block : 1;
}
bool ggml_is_quantized(enum ggml_type type) {
    return type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && type != GGML_TYPE_I32;
}
size_t ggml_type_size(enum ggml_type) { GGML_ABORT("ggml_type_size: not available in ft_mmvq"); }
size_t ggml_nbytes(const struct ggml_tensor *) { GGML_ABORT("ggml_nbytes: not available in ft_mmvq"); }
int64_t ggml_nelements(const struct ggml_tensor *) { GGML_ABORT("ggml_nelements: not available in ft_mmvq"); }
bool ggml_is_contiguous(const struct ggml_tensor *) { GGML_ABORT("ggml_is_contiguous: not available in ft_mmvq"); }
bool ggml_is_contiguously_allocated(const struct ggml_tensor *) { GGML_ABORT("not available in ft_mmvq"); }
bool ggml_are_same_stride(const struct ggml_tensor *, const struct ggml_tensor *) { GGML_ABORT("not available in ft_mmvq"); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { GGML_ABORT("not available in ft_mmvq"); }
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor *) { GGML_ABORT("not available in ft_mmvq"); }
}

static size_t ft_mmvq_type_size(ggml_type type, int * block) {
    switch (type) {
        case GGML_TYPE_Q4_0:    *block = QK4_0;  return sizeof(block_q4_0);
        case GGML_TYPE_Q8_0:    *block = QK8_0;  return sizeof(block_q8_0);
        case GGML_TYPE_Q4_K:    *block = QK_K;   return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K:    *block = QK_K;   return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K:    *block = QK_K;   return sizeof(block_q6_K);
        case GGML_TYPE_IQ3_XXS: *block = QK_K;   return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ4_NL:  *block = QK4_NL; return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ4_XS:  *block = QK_K;   return sizeof(block_iq4_xs);
        default: return 0;
    }
}

extern "C" {

// Device bytes ft_mmvq needs as ``workspace``: the q8_1 activations.
size_t ft_mmvq_workspace(int64_t K, int64_t rows_y) {
    return GGML_PAD(K, MATRIX_ROW_PADDING) / QK8_1 * sizeof(block_q8_1) * rows_y + 4096;
}

// Quantized mat-vec, llama.cpp's MMVQ. Two shapes:
//
//   ids == NULL (dense):   dst[t, :] = W @ x[t, :]              for t < n_tokens (<= 8)
//       W is one ``rows`` x ``K`` matrix; x is f32 [n_tokens, K]; dst f32 [n_tokens, rows].
//   ids != NULL (experts): dst[t, u, :] = W[ids[t, u]] @ x[t, u % ne11, :]
//       W holds n_slots matrices ``slot_stride_bytes`` apart; x is f32 [n_tokens, ne11, K]
//       (ne11 = 1: one row per token shared by every used slot, as for gate/up); ids int32
//       [n_tokens, n_used]; dst f32 [n_tokens, n_used, rows]; n_tokens <= 8.
//
// ``gate`` (optional) fuses the gate projection: a second matrix at the same strides, and
// dst becomes act(x @ gate) * (x @ W) with act the ggml_glu_op ``glu_op`` (SWIGLU = silu).
// Rows of W are ``row_stride_bytes`` apart. Returns 0, -1 unsupported type, -2 bad strides.
int ft_mmvq(const void * W, const void * gate, int glu_op, int type, int64_t K, int64_t rows,
            int64_t row_stride_bytes, int64_t slot_stride_bytes, int64_t n_slots,
            const float * x, int64_t ne11, const int32_t * ids, int64_t n_tokens, int64_t n_used,
            float * dst, void * workspace, size_t workspace_bytes, void * stream_ptr) {
    const ggml_type t = (ggml_type) type;
    int block;
    const size_t ts = ft_mmvq_type_size(t, &block);
    if (ts == 0) {
        return -1;
    }
    if (row_stride_bytes % ts || slot_stride_bytes % ts || K % block || n_tokens > MMVQ_MAX_BATCH_SIZE) {
        return -2;
    }
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t rows_y = ids ? n_tokens * ne11 : n_tokens;
    if (ft_mmvq_workspace(K, rows_y) > workspace_bytes) {
        return -2;
    }
    void * y_q8 = workspace;

    // activations: f32 rows of K, contiguous -> q8_1 rows of k_padded
    if (ids) {
        quantize_row_q8_1_cuda(x, nullptr, y_q8, t, K, /*s01=*/K, /*s02=*/ne11 * K, /*s03=*/ne11 * K * n_tokens,
                               k_padded, ne11, n_tokens, 1, stream);
    } else {
        quantize_row_q8_1_cuda(x, nullptr, y_q8, t, K, /*s01=*/K, /*s02=*/K * n_tokens, /*s03=*/K * n_tokens,
                               k_padded, n_tokens, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate;
    fusion.glu_op = (ggml_glu_op) glu_op;

    const int64_t s01 = row_stride_bytes / ts;
    const int64_t s11 = k_padded / QK8_1;
    if (ids) {
        const int64_t s02 = slot_stride_bytes / ts;
        const int64_t s1 = rows;               // dst: next used slot
        const int64_t s2 = n_used * rows;      // dst: next token
        const int64_t s12 = ne11 * s11;        // y: next token
        // Upstream sends one token to the generic kernel, whose RDNA tuning starves a short K
        // (a 640-wide down projection runs 2.7x slower there on the RX 9060 XT); the
        // dedicated MoE kernel it uses from two tokens on serves one token as well.
        const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
        const uint3 nchannels_y_fd = init_fastdiv_values(ne11);
        switch (t) {
#define FT_MOE(T)                                                                                          \
            case T:                                                                                        \
                mul_mat_vec_q_moe_launch<T>(W, y_q8, ids, fusion, dst, K, nchannels_y_fd, rows,            \
                                            s01, s12, s2, s02, s11, s1, n_tokens, n_used, warp_size,       \
                                            n_used, stream);                                               \
                break;
            FT_MOE(GGML_TYPE_Q4_0)
            FT_MOE(GGML_TYPE_Q8_0)
            FT_MOE(GGML_TYPE_Q4_K)
            FT_MOE(GGML_TYPE_Q5_K)
            FT_MOE(GGML_TYPE_Q6_K)
            FT_MOE(GGML_TYPE_IQ3_XXS)
            FT_MOE(GGML_TYPE_IQ4_NL)
            FT_MOE(GGML_TYPE_IQ4_XS)
#undef FT_MOE
            default:
                return -1;
        }
    } else {
        const int64_t s1 = rows;
        mul_mat_vec_q_switch_type(
            W, t, y_q8, nullptr, fusion, dst, K,
            rows, /*ncols_dst=*/n_tokens, s01, /*stride_col_y=*/s11, /*stride_col_dst=*/s1,
            1, 1, 1, s01 * rows, s11 * n_tokens, s1 * n_tokens,
            1, 1, s01 * rows, s11 * n_tokens, s1 * n_tokens, 0, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

}  // extern "C"
