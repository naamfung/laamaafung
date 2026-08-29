// Fused Q4_0 MMA flash-attention DECODE launcher, env-gated by GGML_CUDA_FATTN_MMA_Q.
//
// Mirrors ggml_cuda_flash_attn_ext_mma_turbo_case in fattn-mma-turbo.cuh but for the
// Q4_0 quantized KV cache. It reuses the f16 MMA device kernel (flash_attn_ext_f16 in
// fattn-mma-f16.cuh) instantiated with type_K/type_V = Q4_0, so the in-kernel load tiles
// dequantize raw block_q4_0 straight into SRAM. launch_fattn is called with
// need_f16_K = need_f16_V = false, so the whole-cache F16 scratch conversion is skipped
// -- the same win the turbo path already gets, now for plain Q4_0.
//
// Difference vs turbo: this path is gated by GGML_CUDA_FATTN_MMA_Q (default 2), which is
// the smallest Q->ne[1] it claims. Below that, and for the F16/other-quant cases, the
// regular kernels still apply.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

// Smallest Q->ne[1] the inline-dequant Q4_0 MMA path claims. 1 stays on the GQA vector
// kernel (the inline dequant costs more than the vector kernel's single-column deficit).
// 0 reproduces upstream behaviour (whole-cache F16 conversion on every call). The default
// matches the op-level sweep measured at kv=54016, q4_0.
static constexpr int FATTN_MMA_Q_MAX_NCOLS1 = 8;

static int ggml_cuda_fattn_mma_q_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_FATTN_MMA_Q");
        if (env == nullptr || *env == '\0') {
            return 2;
        }
        const int v = atoi(env);
        return (v >= 0 && v <= 3) ? v : 2;
    }();
    return mode;
}

// Single source of truth for "does this FA node use the inline-dequant Q4_0 MMA kernel".
// Called both by ggml_cuda_flash_attn_ext_get_alloc_size, which decides whether to reserve
// the F16 scratch, and by the launch path, which decides whether to fill it. If the two
// disagreed the kernel would read a buffer that was never allocated.
static bool ggml_cuda_fattn_mma_use_quantized_kv(const int cc, const ggml_tensor * dst) {
    if (ggml_cuda_fattn_mma_q_mode() == 0) {
        return false;
    }

    // cp.async / synchronous dequant path needs Turing+; on Ada and newer the MMA kernel
    // reads quantized data on tensor cores without a dequant pass, so there the GQA vector
    // or the regular F16 path is better.
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !turing_mma_available(cc) || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Inline dequant runs once per Q tile, so its cost scales with Q->ne[1], while the F16
    // conversion runs once per op. Verify batches (<=8) win; prefill loses.
    if (Q->ne[1] > FATTN_MMA_Q_MAX_NCOLS1) {
        return false;
    }

    if (K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }

    // Only DKQ == DV == 256 is instantiated (see DECL_FATTN_MMA_Q4_0_ALL).
    if (Q->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }

    // Exactly the condition under which the MMA switch picks ncols2 == 8.
    if (!ggml_cuda_fattn_gqa_opt_applies(dst) || Q->ne[2] / K->ne[2] <= 4) {
        return false;
    }

    // cp.async / synchronous dequant needs 16 byte aligned sources and rows.
    for (const ggml_tensor * t : {K, V}) {
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    return true;
}

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_q4_0_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // Q4_0 path is always single-stage synchronous (nstages forced to 0 in the kernel).
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    // Q4_0 never aliases V onto K.
    constexpr bool V_is_K_view = false;

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // need_f16_K = need_f16_V = false: launch_fattn does NOT convert q4_0 blocks to f16;
    // the kernel receives raw quantized KV + the true byte pitch. stream_k = true.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/true, warp_size_host);
}

#define DECL_FATTN_MMA_Q4_0_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                  \
    template void ggml_cuda_flash_attn_ext_mma_q4_0_case                           \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// Decode/verify batches (Q->ne[1] <= FATTN_MMA_Q_MAX_NCOLS1 == 8) reachable (ncols1, ncols2)
// set with turing_mma_available and GQA > 4: ncols2 == 8, ncols1 in {1,2,4}.
#define DECL_FATTN_MMA_Q4_0_ALL(DKQ, DV, tK, tV)        \
    extern DECL_FATTN_MMA_Q4_0_CASE(DKQ, DV, 1, 8, tK, tV); \
    extern DECL_FATTN_MMA_Q4_0_CASE(DKQ, DV, 2, 8, tK, tV); \
    extern DECL_FATTN_MMA_Q4_0_CASE(DKQ, DV, 4, 8, tK, tV);

DECL_FATTN_MMA_Q4_0_ALL(256, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);