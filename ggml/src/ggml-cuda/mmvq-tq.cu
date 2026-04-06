/*
 * Fused mul_mat_vec for TQ4_1S / TQ3_1S weight types.
 *
 * TQ4_1S/TQ3_1S: Pre-rotate activation + centroid lookup (scalar FMA path).
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"

#define MMVQ_TQ_NWARPS 4

// ============================================================================
// Pre-rotate activation to q8_1 format (for TQ4_1S dp4a path)
// ============================================================================

static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// ============================================================================
// TQ4_1S: dp4a path with fixed int8 centroid LUT + q8_1 activation
// ============================================================================

// Fixed int8 centroid table: centroid_i8[i] = round(TQ4_CENTROIDS_WEIGHT[i] * 127 / 2.733)
// Rescale factor to recover float centroids: 2.733 / 127
static constexpr float TQ4_CENTROID_I8_RESCALE = 2.733f / 127.0f;

// Register-based centroid lookup: maps 4 qs bytes (1 uint32) to 2 packed 4× centroid_i8 for dp4a.
// Processes a full uint32 at once, sharing nibble extraction across both byte pairs.
__device__ __forceinline__ void tq4_cents8_reg(uint32_t four_bytes, int &c0, int &c1) {
    // Centroid i8 values packed into 4 registers (little-endian byte order):
    // [-127,-96,-75,-58] [-44,-31,-18,-6] [6,18,31,44] [58,75,96,127]
    constexpr uint32_t CR03 = 0xC6B5A081u;
    constexpr uint32_t CR47 = 0xFAEEE1D4u;
    constexpr uint32_t CR8B = 0x2C1F1206u;
    constexpr uint32_t CRCF = 0x7F604B3Au;

    // Extract all 8 nibbles from 4 bytes at once (shared across both pairs)
    const uint32_t lo = four_bytes & 0x0F0F0F0Fu;
    const uint32_t hi = (four_bytes >> 4) & 0x0F0F0F0Fu;

    // Interleave: bytes 0-1 → sel0 [n0,n1,n2,n3], bytes 2-3 → sel1 [n4,n5,n6,n7]
    const uint32_t sel0 = __byte_perm(lo, hi, 0x5140u);
    const uint32_t sel1 = __byte_perm(lo, hi, 0x7362u);

    // Lookup centroids for sel0 (elements from qs bytes 0-1)
    {
        const uint32_t flo = __byte_perm(CR03, CR47, sel0);
        const uint32_t fhi = __byte_perm(CR8B, CRCF, sel0);
        const uint32_t msb = (sel0 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c0 = (int)__byte_perm(flo, fhi, psel);
    }

    // Lookup centroids for sel1 (elements from qs bytes 2-3)
    {
        const uint32_t flo = __byte_perm(CR03, CR47, sel1);
        const uint32_t fhi = __byte_perm(CR8B, CRCF, sel1);
        const uint32_t msb = (sel1 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c1 = (int)__byte_perm(flo, fhi, psel);
    }
}

static __global__ void mul_mat_vec_tq4_1s_dp4a(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sumf = 0.0f;

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);
        const block_q8_1 * a_blk = &vy_q8[ib];
        const float d_act = __half2float((__half)a_blk->ds.x);

        // Vectorized weight load: 16 bytes as 4× uint32
        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0]; // qs bytes 0-3  → elements 0-7
        const uint32_t w1 = qs32[1]; // qs bytes 4-7  → elements 8-15
        const uint32_t w2 = qs32[2]; // qs bytes 8-11 → elements 16-23
        const uint32_t w3 = qs32[3]; // qs bytes 12-15 → elements 24-31

        // Preload all activation int32s
        const int * a_qs = (const int *)(a_blk->qs);
        const int a0 = a_qs[0], a1 = a_qs[1], a2 = a_qs[2], a3 = a_qs[3];
        const int a4 = a_qs[4], a5 = a_qs[5], a6 = a_qs[6], a7 = a_qs[7];

        // Process full uint32 words: shared nibble extraction saves ~4 ALU ops per word
        int c0, c1;

        tq4_cents8_reg(w0, c0, c1);
        const int s0 = __dp4a(c0, a0, 0);
        const int s1 = __dp4a(c1, a1, 0);

        tq4_cents8_reg(w1, c0, c1);
        const int s2 = __dp4a(c0, a2, 0);
        const int s3 = __dp4a(c1, a3, 0);

        tq4_cents8_reg(w2, c0, c1);
        const int s4 = __dp4a(c0, a4, 0);
        const int s5 = __dp4a(c1, a5, 0);

        tq4_cents8_reg(w3, c0, c1);
        const int s6 = __dp4a(c0, a6, 0);
        const int s7 = __dp4a(c1, a7, 0);

        sumf += d_act * (fd0 * (float)(s0 + s1 + s2 + s3) + fd1 * (float)(s4 + s5 + s6 + s7));
    }

    // Apply centroid int8→float rescale
    sumf *= TQ4_CENTROID_I8_RESCALE;

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);

    if (lane == 0) dst[row] = sumf;
}

// ============================================================================
// Fallback: V8 scalar kernel (pre-rotated float activation)
// ============================================================================

static __global__ void tq_prerotate_activation(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = __float2half(val);
}

static __global__ void mul_mat_vec_tq4_1s_fused(
        const void  * __restrict__ vx,
        const half  * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    // Byte→half2 LUT: precompute all 256 (lo_nibble, hi_nibble) centroid pairs.
    // One shared mem read per byte instead of two divergent reads.
    __shared__ half2 s_lut_h2[256];
    {
        const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
        #pragma unroll
        for (int i = tid; i < 256; i += MMVQ_TQ_NWARPS * WARP_SIZE) {
            s_lut_h2[i] = make_half2(
                __float2half(TQ4_CENTROIDS_WEIGHT[i & 0xF]),
                __float2half(TQ4_CENTROIDS_WEIGHT[(i >> 4) & 0xF]));
        }
    }
    __syncthreads();

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    // Strided block processing: each lane handles different blocks (coalesced weight loads).
    // Within each block, process element pairs via half2 for 2x arithmetic density.
    // Factor half-block scale out of inner loop: accumulate act*centroid in half2,
    // multiply by d once per half-block. Eliminates 16 __hmul2 per block.
    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_1s * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);
        const half * act = vy_rot + ib * QK_TQ4_1S;

        // Vectorized weight load: 16 bytes as 4× uint32 (vs 16 byte loads)
        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0];
        const uint32_t w1 = qs32[1];
        const uint32_t w2 = qs32[2];
        const uint32_t w3 = qs32[3];

        // Vectorized activation load: 64 bytes as 4× int4 / 128-bit (vs 16× half2)
        const int4 * act128 = (const int4 *)act;
        const int4 a_vec0 = act128[0]; // halfs 0-7
        const int4 a_vec1 = act128[1]; // halfs 8-15
        const int4 a_vec2 = act128[2]; // halfs 16-23
        const int4 a_vec3 = act128[3]; // halfs 24-31

        half2 hsum0 = make_half2(__float2half(0.0f), __float2half(0.0f));
        half2 hsum1 = make_half2(__float2half(0.0f), __float2half(0.0f));

        // First half: w0 (qs bytes 0-3) × activations 0-7
        {
            const half2 * a_h2 = (const half2 *)&a_vec0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const half2 c = s_lut_h2[(w0 >> (j * 8)) & 0xFF];
                hsum0 = __hfma2(a_h2[j], c, hsum0);
            }
        }
        // w1 (qs bytes 4-7) × activations 8-15
        {
            const half2 * a_h2 = (const half2 *)&a_vec1;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const half2 c = s_lut_h2[(w1 >> (j * 8)) & 0xFF];
                hsum0 = __hfma2(a_h2[j], c, hsum0);
            }
        }
        // Second half: w2 (qs bytes 8-11) × activations 16-23
        {
            const half2 * a_h2 = (const half2 *)&a_vec2;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const half2 c = s_lut_h2[(w2 >> (j * 8)) & 0xFF];
                hsum1 = __hfma2(a_h2[j], c, hsum1);
            }
        }
        // w3 (qs bytes 12-15) × activations 24-31
        {
            const half2 * a_h2 = (const half2 *)&a_vec3;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const half2 c = s_lut_h2[(w3 >> (j * 8)) & 0xFF];
                hsum1 = __hfma2(a_h2[j], c, hsum1);
            }
        }

        // Multiply by half-block scale once, convert to float
        sum += (__half2float(hsum0.x) + __half2float(hsum0.y)) * fd0
             + (__half2float(hsum1.x) + __half2float(hsum1.y)) * fd1;
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

static __global__ void mul_mat_vec_tq3_1s_fused(
        const void  * __restrict__ vx,
        const half  * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    // Shared memory LUT avoids constant memory serialization (divergent idx across lanes)
    __shared__ float s_lut[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        s_lut[threadIdx.x] = TQ3_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = __half2float(vy_rot[ib * QK_TQ3_0 + lane]);
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        sum += act * s_lut[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// Dispatch: try dp4a, fall back to V8 scalar
// ============================================================================

static half * d_act_buf = nullptr;
static size_t  d_act_buf_size = 0;
static block_q8_1 * d_q8_1_buf = nullptr;
static size_t  d_q8_1_buf_size = 0;

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0,
                               const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    // Graph-safe scratch allocation
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream, &capture_status);

    if (capture_status == cudaStreamCaptureStatusNone) {
        const size_t act_needed = ncols_x * sizeof(half);
        if (act_needed > d_act_buf_size) {
            if (d_act_buf) cudaFree(d_act_buf);
            cudaMalloc(&d_act_buf, act_needed);
            d_act_buf_size = act_needed;
        }

        const int n_blocks = ncols_x / 32;
        const size_t q8_1_needed = n_blocks * sizeof(block_q8_1);
        if (q8_1_needed > d_q8_1_buf_size) {
            if (d_q8_1_buf) cudaFree(d_q8_1_buf);
            cudaMalloc(&d_q8_1_buf, q8_1_needed);
            d_q8_1_buf_size = q8_1_needed;
        }
    }

    if (src0->type == GGML_TYPE_TQ4_1S) {
        // TQ4_1S: dp4a path with q8_1 pre-rotated activation
        GGML_ASSERT(d_q8_1_buf != nullptr);
        const int n_blocks = ncols_x / 32;

        // Phase 1: Pre-rotate + q8_1 quantize
        {
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_blocks + wpb - 1) / wpb);
            tq_prerotate_q8_1<<<grid, block, 0, stream>>>(src1_d, d_q8_1_buf, ncols_x);
        }

        // Phase 2: dp4a mmvq
        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
            mul_mat_vec_tq4_1s_dp4a<<<grid, block, 0, stream>>>(src0_d, d_q8_1_buf, dst_d, ncols_x, nrows_x);
        }
    } else {
        // TQ3_1S: V8 scalar path with half pre-rotated activation
        GGML_ASSERT(d_act_buf != nullptr);
        {
            const int n_blocks = ncols_x / 32;
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_blocks + wpb - 1) / wpb);
            tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, d_act_buf, ncols_x);
        }
        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
            mul_mat_vec_tq3_1s_fused<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
        }
    }
}


// ============================================================================
// Load-time conversion: TQ4_1S → q8_0 (opt-in via GGML_TQ_CONVERT_Q8=1)
// ============================================================================

static __global__ void k_convert_tq4_1s_to_q8_0(
        const block_tq4_1s * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;
    const int lane = threadIdx.x;
    const block_tq4_1s * blk = &src[block_idx];

    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    val *= TQ_WEIGHT_SIGNS[lane];

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) dst[block_idx].d = __float2half(d);
}

void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_TQ4_1S == 0);
    const int n_blocks = n_elements / QK_TQ4_1S;
    const int wpb = 4;
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);
    k_convert_tq4_1s_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_tq4_1s *)src_tq4, (block_q8_0 *)dst_q8, n_blocks);
}
