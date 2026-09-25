// FreeToken glue for the vendored llama.cpp MMQ (see UPSTREAM.md).
//
// Two jobs:
//   1. Define the handful of runtime symbols the MMQ headers declare but ggml-cuda.cu and
//      ggml.c would normally provide (device info, error reporting, the scratch pool).
//   2. Export a C entry point, ft_mmq_moe(), that does what ggml_cuda_mul_mat_q() does for
//      GGML_OP_MUL_MAT_ID: group the rows by expert (mm_ids_helper), quantize the
//      activations to the MMQ q8_1 layout, and run mul_mat_q_case<type>. Pointers and strides
//      come from the caller (torch tensors); no ggml_tensor is ever built.
//
// Scratch memory is a caller-provided device buffer handed out bump-allocator style for the
// duration of one call (ft_mmq_moe_workspace() says how big), so the kernels never allocate.
// The instantiations of mul_mat_q_case live in ft_mmq_inst_<type>.cu, generated at build
// time, one translation unit per type so they compile in parallel.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/mmid.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

// ------------------------------------------------------------------------------------------
// What ggml.c provides
// ------------------------------------------------------------------------------------------

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ft_mmq: %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

extern "C" void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    if (level < GGML_LOG_LEVEL_WARN) {
        return;
    }
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

// ------------------------------------------------------------------------------------------
// What ggml-cuda.cu provides
// ------------------------------------------------------------------------------------------

// ggml_cuda_parse_id, verbatim: gcnArchName -> the cc the MMQ configs are keyed by.
static int ft_parse_arch(const char * devName) {
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    } else {
        return archNum;
    }
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    if (archLen > 8) {
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int) strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int) strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        archMinor = (int) strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';
        archMajor = (int) strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;
    return archNum;
}

static ggml_cuda_device_info ft_cuda_init() {
    ggml_cuda_device_info info = {};
    CUDA_CHECK(cudaGetDeviceCount(&info.physical_device_count));
    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);
    info.device_count = info.physical_device_count;
    for (int id = 0; id < info.device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        auto & d = info.devices[id];
        d.physical_device = id;
        d.physical_share_count = 1;
        d.virtual_index = 0;
        d.integrated = false;
        d.vmm = false;
        d.total_vram = prop.totalGlobalMem;
        d.nsm = prop.multiProcessorCount;
        d.smpb = prop.sharedMemPerBlock;
        d.smpbo = prop.sharedMemPerBlock;
        d.warp_size = prop.warpSize;
        d.supports_cooperative_launch = false;
        d.cc = ft_parse_arch(prop.gcnArchName);
    }
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ft_cuda_init();
    return info;
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_set_device(int device) {
    int current;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "ft_mmq: CUDA/HIP error: %s\n  in %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    abort();
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

// The scratch pool: one caller-provided device buffer per call, handed out bump-allocator
// style. Nothing is freed inside a call; the next call starts over.
namespace {
struct ft_arena {
    char * base = nullptr;
    size_t size = 0;
    size_t used = 0;
};
thread_local ft_arena g_arena;

void * arena_alloc(size_t size) {
    const size_t start = (g_arena.used + 255) & ~size_t(255);
    if (start + size > g_arena.size) {
        GGML_ABORT("ft_mmq workspace too small: need %zu more bytes past %zu of %zu",
                   size, start, g_arena.size);
    }
    g_arena.used = start + size;
    return g_arena.base + start;
}

struct ft_arena_pool : ggml_cuda_pool {
    void * alloc(size_t size, size_t * actual_size) override {
        *actual_size = size;
        return arena_alloc(size);
    }
    void free(void *, size_t) override {}
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::make_unique<ft_arena_pool>();
}

// ------------------------------------------------------------------------------------------
// Entry points
// ------------------------------------------------------------------------------------------

// Instantiated in ft_mmq_inst_<type>.cu; a type without one is reported, not linked.
static bool ft_dispatch(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
#define FT_CASE(T) case T: mul_mat_q_case<T>(ctx, args, stream); return true;
        FT_CASE(GGML_TYPE_Q4_0)
        FT_CASE(GGML_TYPE_Q8_0)
        FT_CASE(GGML_TYPE_Q4_K)
        FT_CASE(GGML_TYPE_Q5_K)
        FT_CASE(GGML_TYPE_Q6_K)
        FT_CASE(GGML_TYPE_IQ3_XXS)
        FT_CASE(GGML_TYPE_IQ4_NL)
        FT_CASE(GGML_TYPE_IQ4_XS)
#undef FT_CASE
        default:
            return false;
    }
}

// Block bytes and elements per block of the types ft_dispatch serves.
static bool ft_type_geometry(ggml_type type, size_t * type_size, int * block) {
    switch (type) {
        case GGML_TYPE_Q4_0:    *type_size = sizeof(block_q4_0);    *block = QK4_0;  return true;
        case GGML_TYPE_Q8_0:    *type_size = sizeof(block_q8_0);    *block = QK8_0;  return true;
        case GGML_TYPE_Q4_K:    *type_size = sizeof(block_q4_K);    *block = QK_K;   return true;
        case GGML_TYPE_Q5_K:    *type_size = sizeof(block_q5_K);    *block = QK_K;   return true;
        case GGML_TYPE_Q6_K:    *type_size = sizeof(block_q6_K);    *block = QK_K;   return true;
        case GGML_TYPE_IQ3_XXS: *type_size = sizeof(block_iq3_xxs); *block = QK_K;   return true;
        case GGML_TYPE_IQ4_NL:  *type_size = sizeof(block_iq4_nl);  *block = QK4_NL; return true;
        case GGML_TYPE_IQ4_XS:  *type_size = sizeof(block_iq4_xs);  *block = QK_K;   return true;
        default: return false;
    }
}

static size_t ft_q8_bytes(ggml_type type, int64_t K, int64_t rows_y, int64_t ne11) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    return rows_y * k_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
           ggml_cuda_mmq_get_J_max(type, /*fallback=*/true, cc, ne11) * sizeof(block_q8_1_mmq) +
           ggml_cuda_mmq_get_J_max(type, /*fallback=*/false, cc, ne11) * sizeof(block_q8_1_mmq);
}

extern "C" {

// 1 if ft_mmq_moe serves ``type``.
int ft_mmq_supports(int type) {
    size_t ts;
    int block;
    return ft_type_geometry((ggml_type) type, &ts, &block) ? 1 : 0;
}

// Device bytes ft_mmq_moe needs as ``workspace`` for this shape.
size_t ft_mmq_moe_workspace(int type, int64_t K, int64_t n_slots, int64_t n_tokens, int64_t n_used, int64_t ne11) {
    const int id = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int64_t rows_y = n_tokens * n_used;
    size_t total = 0;
    total += 2 * GGML_PAD(rows_y * sizeof(int32_t), 256);             // ids_src1, ids_dst
    total += GGML_PAD((n_slots + 1) * sizeof(int32_t), 256);          // expert_bounds
    total += GGML_PAD(ft_q8_bytes((ggml_type) type, K, rows_y, ne11), 256);
    total += GGML_PAD((size_t) nsm * 2 * 256 * 256 * sizeof(float), 256);  // stream-k fixup, generous
    return total + 4096;
}

// ``dst[t, u, :] = W[ids[t, u]] @ x[t, u % ne11, :]`` for every token t and used slot u.
//
//   W        n_slots expert matrices of ``rows`` x ``K`` in ggml ``type`` blocks, the expert
//            ``e`` at ``W + e * slot_stride_bytes``, its rows ``row_stride_bytes`` apart
//   x        f32 activations; ``ne11`` rows per token (1: the same row for every used slot,
//            as for gate/up; n_used: one row per slot, as for down), rows of K floats
//            ``K`` apart, tokens ``x_token_stride`` floats apart
//   ids      int32 [n_tokens, n_used] slot ids
//   dst      f32 [n_tokens, n_used, rows], contiguous
//   n_experts_eff  how many distinct experts the routing can hit (the tile size is picked
//            against the average rows per expert, as upstream does on RDNA)
//
// Returns 0, or a negative code: -1 unsupported type, -2 misaligned strides.
int ft_mmq_moe(const void * W, int type, int64_t K, int64_t rows, int64_t row_stride_bytes,
               int64_t slot_stride_bytes, int64_t n_slots, int64_t n_experts_eff,
               const float * x, int64_t ne11, int64_t x_token_stride,
               const int32_t * ids, int64_t n_tokens, int64_t n_used,
               float * dst, void * workspace, size_t workspace_bytes, void * stream_ptr) {
    const ggml_type t = (ggml_type) type;
    size_t ts;
    int block;
    if (!ft_type_geometry(t, &ts, &block)) {
        return -1;
    }
    if (row_stride_bytes % ts || slot_stride_bytes % ts || K % block) {
        return -2;
    }
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    g_arena = {(char *) workspace, workspace_bytes, 0};

    const int64_t rows_y = n_tokens * n_used;
    const int64_t k_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    int32_t * ids_src1 = (int32_t *) arena_alloc(rows_y * sizeof(int32_t));
    int32_t * ids_dst = (int32_t *) arena_alloc(rows_y * sizeof(int32_t));
    int32_t * expert_bounds = (int32_t *) arena_alloc((n_slots + 1) * sizeof(int32_t));

    // gate/up rows are shared by every used slot of a token: quantize each token once
    const bool dedup_bcast = ne11 == 1 && n_used > 1;
    ggml_cuda_launch_mm_ids_helper(ids, ids_src1, ids_dst, expert_bounds, (int) n_slots, (int) n_tokens,
                                   (int) n_used, (int) ne11, /*si1=*/(int) n_used, /*sis1=*/(int) ne11,
                                   dedup_bcast, stream);
    CUDA_CHECK(cudaGetLastError());

    void * y_q8 = arena_alloc(ft_q8_bytes(t, K, rows_y, ne11));
    if (dedup_bcast) {
        quantize_scatter_mmq_q8_1_cuda(x, ids_src1, y_q8, t, K, /*stride_token=*/x_token_stride, k_padded,
                                       n_tokens, rows_y, (int) n_used, stream);
    } else {
        quantize_mmq_q8_1_cuda(x, ids_src1, y_q8, t, K, /*s01=*/K, /*s02=*/x_token_stride,
                               /*s03=*/x_token_stride * n_tokens, k_padded, rows_y, 1, 1, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    const int64_t s12 = ne11 * k_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = n_tokens * s12;
    const int64_t s01 = row_stride_bytes / ts;
    const int64_t s02 = slot_stride_bytes / ts;
    const int64_t s1 = rows;
    const int64_t s2 = n_used * rows;

    int64_t ncols_opt = n_tokens;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (rows_y + n_experts_eff - 1) / n_experts_eff;
    }

    const mmq_args args = {
        (const char *) W, t, (const int *) y_q8, ids_dst, expert_bounds, dst, nullptr,
        K, rows, rows_y, s01, rows_y, s1,
        n_slots, n_slots, s02, s12, s2,
        1, 1, s02 * n_slots, s13, s2 * n_tokens,
        n_tokens, ncols_opt};

    ggml_backend_cuda_context ctx(id);
    if (!ft_dispatch(ctx, args, stream)) {
        return -1;
    }
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

}  // extern "C"
