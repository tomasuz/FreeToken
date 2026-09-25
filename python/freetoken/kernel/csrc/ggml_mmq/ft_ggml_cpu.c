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

// For weight type ``type``: its row dot product, the activation type it pairs with, that
// type's row quantizer and its block (elements, bytes). Returns 0, or -1 when the type is
// not served here.
int ft_ggml_cpu_lookup(int type, void ** vec_dot, int * vec_dot_type, void ** quant,
                       int64_t * act_block, int64_t * act_block_bytes) {
    ft_vec_dot_t dot = NULL;
    int vdt = GGML_TYPE_Q8_K;
    switch ((enum ggml_type) type) {
        case GGML_TYPE_Q4_0:    dot = ggml_vec_dot_q4_0_q8_0;    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q5_0:    dot = ggml_vec_dot_q5_0_q8_0;    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q8_0:    dot = ggml_vec_dot_q8_0_q8_0;    vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_IQ4_NL:  dot = ggml_vec_dot_iq4_nl_q8_0;  vdt = GGML_TYPE_Q8_0; break;
        case GGML_TYPE_Q2_K:    dot = ggml_vec_dot_q2_K_q8_K;    break;
        case GGML_TYPE_Q3_K:    dot = ggml_vec_dot_q3_K_q8_K;    break;
        case GGML_TYPE_Q4_K:    dot = ggml_vec_dot_q4_K_q8_K;    break;
        case GGML_TYPE_Q5_K:    dot = ggml_vec_dot_q5_K_q8_K;    break;
        case GGML_TYPE_Q6_K:    dot = ggml_vec_dot_q6_K_q8_K;    break;
        case GGML_TYPE_IQ2_XXS: dot = ggml_vec_dot_iq2_xxs_q8_K; break;
        case GGML_TYPE_IQ2_XS:  dot = ggml_vec_dot_iq2_xs_q8_K;  break;
        case GGML_TYPE_IQ2_S:   dot = ggml_vec_dot_iq2_s_q8_K;   break;
        case GGML_TYPE_IQ3_XXS: dot = ggml_vec_dot_iq3_xxs_q8_K; break;
        case GGML_TYPE_IQ3_S:   dot = ggml_vec_dot_iq3_s_q8_K;   break;
        case GGML_TYPE_IQ4_XS:  dot = ggml_vec_dot_iq4_xs_q8_K;  break;
        default: return -1;
    }
    *vec_dot = (void *) dot;
    *vec_dot_type = vdt;
    if (vdt == GGML_TYPE_Q8_0) {
        *quant = (void *) quantize_row_q8_0;
        *act_block = QK8_0;
        *act_block_bytes = sizeof(block_q8_0);
    } else {
        *quant = (void *) quantize_row_q8_K;
        *act_block = QK_K;
        *act_block_bytes = sizeof(block_q8_K);
    }
    return 0;
}
