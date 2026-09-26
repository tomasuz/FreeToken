// FreeToken entry point for the vendored llama.cpp CPU dot products (see UPSTREAM.md).
//
// ggml-cpu/quants.c, ggml-cpu/arch/x86/quants.c and ggml-quants.c are compiled as they are;
// this file supplies the few ggml.c symbols they reference and one lookup,
// ft_ggml_cpu_lookup(), that hands the CPU MoE executor (csrc/cpu_moe) the row dot product
// of a weight type, the type its activations must be quantized to, and that quantizer --
// the same trio ggml_get_type_traits_cpu() answers inside llama.cpp.

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/arch-fallback.h"  // the *_generic names an arch serves itself
#include "ggml-quants.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

__attribute__((constructor)) static void ft_ggml_cpu_tables(void) {
    for (int i = 0; i < (1 << 16); ++i) {
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32((ggml_fp16_t) i);
    }
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "%s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

// referenced by the weight quantizers of ggml-quants.c, which FreeToken never calls
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void) type; (void) ne; GGML_ABORT("ggml_row_size: not available in ft_ggml_cpu"); }
size_t ggml_type_size(enum ggml_type type) { (void) type; GGML_ABORT("ggml_type_size: not available in ft_ggml_cpu"); }
const char * ggml_type_name(enum ggml_type type) { (void) type; return "?"; }

typedef void (*ft_vec_dot_t)(int n, float * s, size_t bs, const void * x, size_t bx, const void * y, size_t by, int nrc);
typedef void (*ft_quant_t)(const float * x, void * y, int64_t k);

static int ft_lookup(int type, int generic, void ** vec_dot, int * vec_dot_type, void ** quant,
                     int64_t * act_block, int64_t * act_block_bytes);

// For weight type ``type``: its row dot product, the activation type it pairs with, that
// type's row quantizer and its block (elements, bytes). Returns 0, or -1 when the type is
// not served here.
int ft_ggml_cpu_lookup(int type, void ** vec_dot, int * vec_dot_type, void ** quant,
                       int64_t * act_block, int64_t * act_block_bytes) {
    return ft_lookup(type, 0, vec_dot, vec_dot_type, quant, act_block, act_block_bytes);
}

// The same lookup over llama.cpp's portable C versions: the reference the SIMD ones are
// tested against, on any host.
int ft_ggml_cpu_lookup_generic(int type, void ** vec_dot, int * vec_dot_type, void ** quant,
                               int64_t * act_block, int64_t * act_block_bytes) {
    return ft_lookup(type, 1, vec_dot, vec_dot_type, quant, act_block, act_block_bytes);
}

#define FT_DOT(name) (generic ? (ft_vec_dot_t) name##_generic : (ft_vec_dot_t) name)

static int ft_lookup(int type, int generic, void ** vec_dot, int * vec_dot_type, void ** quant,
                     int64_t * act_block, int64_t * act_block_bytes) {
    ft_vec_dot_t dot = NULL;
    int vdt = GGML_TYPE_Q8_K;
    switch ((enum ggml_type) type) {
        case GGML_TYPE_Q4_0:    dot = FT_DOT(ggml_vec_dot_q4_0_q8_0);    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q5_0:    dot = FT_DOT(ggml_vec_dot_q5_0_q8_0);    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q8_0:    dot = FT_DOT(ggml_vec_dot_q8_0_q8_0);    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_IQ4_NL:  dot = FT_DOT(ggml_vec_dot_iq4_nl_q8_0);  vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q2_K:    dot = FT_DOT(ggml_vec_dot_q2_K_q8_K);    break;
        case GGML_TYPE_Q3_K:    dot = FT_DOT(ggml_vec_dot_q3_K_q8_K);    break;
        case GGML_TYPE_Q4_K:    dot = FT_DOT(ggml_vec_dot_q4_K_q8_K);    break;
        case GGML_TYPE_Q5_K:    dot = FT_DOT(ggml_vec_dot_q5_K_q8_K);    break;
        case GGML_TYPE_Q6_K:    dot = FT_DOT(ggml_vec_dot_q6_K_q8_K);    break;
        case GGML_TYPE_IQ2_XXS: dot = FT_DOT(ggml_vec_dot_iq2_xxs_q8_K); break;
        case GGML_TYPE_IQ2_XS:  dot = FT_DOT(ggml_vec_dot_iq2_xs_q8_K);  break;
        case GGML_TYPE_IQ2_S:   dot = FT_DOT(ggml_vec_dot_iq2_s_q8_K);   break;
        case GGML_TYPE_IQ3_XXS: dot = FT_DOT(ggml_vec_dot_iq3_xxs_q8_K); break;
        case GGML_TYPE_IQ3_S:   dot = FT_DOT(ggml_vec_dot_iq3_s_q8_K);   break;
        case GGML_TYPE_IQ4_XS:  dot = FT_DOT(ggml_vec_dot_iq4_xs_q8_K);  break;
        default: return -1;
    }
    *vec_dot = (void *) dot;
    *vec_dot_type = vdt;
    if (vdt == GGML_TYPE_Q8_0) {
        *quant = generic ? (void *) quantize_row_q8_0_generic : (void *) quantize_row_q8_0;
        *act_block = QK8_0;
        *act_block_bytes = sizeof(block_q8_0);
    } else {
        *quant = generic ? (void *) quantize_row_q8_K_generic : (void *) quantize_row_q8_K;
        *act_block = QK_K;
        *act_block_bytes = sizeof(block_q8_K);
    }
    return 0;
}

// Quantize ``k`` floats (a whole number of blocks) to weight type ``type`` with llama.cpp's
// reference quantizer, for tests. Returns 0, or -1 for a type without one that needs no
// set-up (the i-quants below IQ4 need their grids initialised first).
int ft_ggml_quantize_ref(int type, const float * x, void * y, int64_t k) {
    switch ((enum ggml_type) type) {
        case GGML_TYPE_Q4_0:   quantize_row_q4_0_ref(x, (block_q4_0 *) y, k); return 0;
        case GGML_TYPE_Q5_0:   quantize_row_q5_0_ref(x, (block_q5_0 *) y, k); return 0;
        case GGML_TYPE_Q8_0:   quantize_row_q8_0_ref(x, (block_q8_0 *) y, k); return 0;
        case GGML_TYPE_Q2_K:   quantize_row_q2_K_ref(x, (block_q2_K *) y, k); return 0;
        case GGML_TYPE_Q3_K:   quantize_row_q3_K_ref(x, (block_q3_K *) y, k); return 0;
        case GGML_TYPE_Q4_K:   quantize_row_q4_K_ref(x, (block_q4_K *) y, k); return 0;
        case GGML_TYPE_Q5_K:   quantize_row_q5_K_ref(x, (block_q5_K *) y, k); return 0;
        case GGML_TYPE_Q6_K:   quantize_row_q6_K_ref(x, (block_q6_K *) y, k); return 0;
        case GGML_TYPE_IQ4_NL: quantize_row_iq4_nl_ref(x, (block_iq4_nl *) y, k); return 0;
        case GGML_TYPE_IQ4_XS: quantize_row_iq4_xs_ref(x, (block_iq4_xs *) y, k); return 0;
        default: return -1;
    }
}
