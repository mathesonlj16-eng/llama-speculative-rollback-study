#pragma once
// Verbatim copy (extracted by script) of the production Q8_0 kernel gemv_prod from bi_n1_ctrl.cu (= in-tree mul_mat_bi_fast_n1_q8_0).
#include <cuda_fp16.h>
#include <cstdint>
struct block_q8_0_t { half d; int8_t qs[32]; };
struct block_q8_1_mmq_t { union { float d4[4]; half2 ds4[4]; half d2s6[8]; }; int8_t qs[128]; };
static_assert(sizeof(block_q8_0_t) == 34 && sizeof(block_q8_1_mmq_t) == 144, "sizes");

// ---------------- production kernel (copy of in-tree mul_mat_bi_fast_n1_q8_0, nwarps = 2) ----------------
template <int NW>
__global__ void __launch_bounds__(NW*32, 1) gemv_prod(
        const char * __restrict__ W, const int64_t row_stride, const block_q8_1_mmq_t * __restrict__ y,
        float * __restrict__ dst, const int nrows, const int nb) {
    __shared__ int4  sbuf[NW][69];
    __shared__ float4 st[NW][8];
    __shared__ float4 sd[NW][8];
    const int row = blockIdx.x*NW + threadIdx.y;
    if (row >= nrows) return;
    const int lane = threadIdx.x, wy = threadIdx.y;
    const int4 * wr = (const int4 *) (W + (int64_t) row*row_stride);
    const int nchunks = nb / 32;
    int4 pf[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) { const int idx = lane + 32*i; if (idx < 68) pf[i] = __ldg(wr + idx); }
    const int * s32 = (const int *) sbuf[wy];
    const int w0 = 8*lane + lane/2, sh = (lane & 1) * 16;
    float s = 0.0f;
    for (int ch = 0; ch < nchunks; ++ch) {
#pragma unroll
        for (int i = 0; i < 3; ++i) { const int idx = lane + 32*i; if (idx < 68) sbuf[wy][idx] = pf[i]; }
        if (ch + 1 < nchunks) {
#pragma unroll
            for (int i = 0; i < 3; ++i) { const int idx = lane + 32*i; if (idx < 68) pf[i] = __ldg(wr + 68*(ch + 1) + idx); }
        }
        const int b = 32*ch + lane;
        const block_q8_1_mmq_t * by = y + (b >> 2);
        const int4 ya = __ldg((const int4 *) (by->qs + 32*(b & 3)) + 0);
        const int4 yb = __ldg((const int4 *) (by->qs + 32*(b & 3)) + 1);
        const float dB = __ldg(&by->d4[b & 3]);
        __syncwarp();
        int w[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) w[i] = s32[w0 + i];
        const uint16_t dbits = uint16_t(uint32_t(w[0]) >> sh);
        int v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = (int) __funnelshift_rc((unsigned) w[i], (unsigned) w[i + 1], 16 + sh);
        int c = __dp4a(v[0], ya.x, 0);
        c = __dp4a(v[1], ya.y, c); c = __dp4a(v[2], ya.z, c); c = __dp4a(v[3], ya.w, c);
        c = __dp4a(v[4], yb.x, c); c = __dp4a(v[5], yb.y, c); c = __dp4a(v[6], yb.z, c); c = __dp4a(v[7], yb.w, c);
        const float t = __fmul_rn((float) c, __half2float(__ushort_as_half(dbits)));
        ((float *) st[wy])[lane] = t;
        ((float *) sd[wy])[lane] = dB;
        __syncwarp();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float4 tq = st[wy][q], dq = sd[wy][q];
            s = __fmaf_rn(tq.x, dq.x, s); s = __fmaf_rn(tq.y, dq.y, s);
            s = __fmaf_rn(tq.z, dq.z, s); s = __fmaf_rn(tq.w, dq.w, s);
        }
        __syncwarp();
    }
    if (lane == 0) dst[row] = s;
}

