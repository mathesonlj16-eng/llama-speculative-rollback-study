// Controls for the fast N=1 Q8_0 kernel (Arm A follow-up):
//  - negative controls (deliberately wrong arithmetic must FAIL the exact compare),
//  - adversarial inputs over many seeds, exact byte compare (memcmp), NaN/Inf counts,
//  - fp64 reference on the quantized operands (correctness, not just invariance).
//
// Reference = ggml MUL_MAT on CUDA0, batch-invariant flag on, N in NSET.
// Linked against build/pr-bi   : every N (including N=1) is forced MMQ.
// Linked against build/mmq-n1  : N=1 goes to the in-tree fast path, N>1 to forced MMQ,
//                                so "ref self-invariance" then tests the in-tree kernel.
//
// Build: nvcc -O3 -use_fast_math -arch=sm_75 (same device flags as ggml-cuda).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define FULL 0xffffffffu

struct block_q8_0_t { half d; int8_t qs[32]; };
struct block_q8_1_mmq_t { union { float d4[4]; half2 ds4[4]; half d2s6[8]; }; int8_t qs[128]; };
static_assert(sizeof(block_q8_0_t) == 34 && sizeof(block_q8_1_mmq_t) == 144, "sizes");

// ---------------- quantizers ----------------
// (Q) copy of quantize_mmq_q8_1<D4> math (quantize.cu:456-570 @ 3f25ea199), one column.
__global__ void quant_mmq_d4(const float * __restrict__ x, void * __restrict__ vy, const int64_t ne00, const int64_t ne0) {
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) return;
    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq_t * y = (block_q8_1_mmq_t *) vy;
    const int64_t k_block = i0 / 128, iqs = i0 % 128;
    const float4 xi = i0 < ne00 ? x4[i0/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y)); amax = fmaxf(amax, fabsf(xi.z)); amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, offset, 32));
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv); q.y = roundf(xi.y*d_inv); q.z = roundf(xi.z*d_inv); q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;
    char4 * yqs4 = (char4 *) y[k_block].qs;
    yqs4[iqs/4] = q;
    if (iqs % 32 == 0) y[k_block].d4[iqs/32] = d;
}

// (control d) MMVQ-style quantizer math (quantize.cu:85-100: d = amax/127, q = round(x/d), d stored as half),
// written into the same MMQ layout so that only the quantizer arithmetic differs.
__global__ void quant_mmvq_style(const float * __restrict__ x, void * __restrict__ vy, const int64_t ne00, const int64_t ne0) {
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) return;
    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq_t * y = (block_q8_1_mmq_t *) vy;
    const int64_t k_block = i0 / 128, iqs = i0 % 128;
    const float4 xi = i0 < ne00 ? x4[i0/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y)); amax = fmaxf(amax, fabsf(xi.z)); amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, offset, 32));
    const float d = amax / 127.0f;
    char4 q;
    q.x = amax == 0.0f ? 0 : roundf(xi.x / d); q.y = amax == 0.0f ? 0 : roundf(xi.y / d);
    q.z = amax == 0.0f ? 0 : roundf(xi.z / d); q.w = amax == 0.0f ? 0 : roundf(xi.w / d);
    char4 * yqs4 = (char4 *) y[k_block].qs;
    yqs4[iqs/4] = q;
    if (iqs % 32 == 0) y[k_block].d4[iqs/32] = __half2float(__float2half(d));
}

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

// ---------------- control kernels: same loads/integer part, only the float program differs ----------------
// MODE 1 (a): (dA*dB)*Cf, i.e. s = fma(dA*dB, Cf, s)      [dp4a vec_dot_q8_0_q8_1_impl association]
// MODE 2 (b): unfused s = s + (Cf*dA)*dB (FMUL, FMUL, FADD)
// MODE 3 (c): 2-way split-K: even blocks -> s0, odd -> s1 (each MMQ-order fma), s = s0 + s1
// MODE 4 (e): extra: fold in descending block order within each 32-block chunk
template <int MODE>
__global__ void __launch_bounds__(64, 1) gemv_ctrl(
        const char * __restrict__ W, const int64_t row_stride, const block_q8_1_mmq_t * __restrict__ y,
        float * __restrict__ dst, const int nrows, const int nb) {
    constexpr int NW = 2;
    __shared__ int4  sbuf[NW][69];
    __shared__ float sc[NW][32], sa[NW][32], sd[NW][32];
    const int row = blockIdx.x*NW + threadIdx.y;
    if (row >= nrows) return;
    const int lane = threadIdx.x, wy = threadIdx.y;
    const int4 * wr = (const int4 *) (W + (int64_t) row*row_stride);
    const int nchunks = nb / 32;
    const int * s32 = (const int *) sbuf[wy];
    const int w0 = 8*lane + lane/2, sh = (lane & 1) * 16;
    float s = 0.0f, s0 = 0.0f, s1 = 0.0f;
    for (int ch = 0; ch < nchunks; ++ch) {
        for (int i = 0; i < 3; ++i) { const int idx = lane + 32*i; if (idx < 68) sbuf[wy][idx] = __ldg(wr + 68*ch + idx); }
        const int b = 32*ch + lane;
        const block_q8_1_mmq_t * by = y + (b >> 2);
        const int4 ya = __ldg((const int4 *) (by->qs + 32*(b & 3)) + 0);
        const int4 yb = __ldg((const int4 *) (by->qs + 32*(b & 3)) + 1);
        const float dB = __ldg(&by->d4[b & 3]);
        __syncwarp();
        int w[9];
        for (int i = 0; i < 9; ++i) w[i] = s32[w0 + i];
        const uint16_t dbits = uint16_t(uint32_t(w[0]) >> sh);
        int v[8];
        for (int i = 0; i < 8; ++i) v[i] = (int) __funnelshift_rc((unsigned) w[i], (unsigned) w[i + 1], 16 + sh);
        int c = 0;
        c = __dp4a(v[0], ya.x, c); c = __dp4a(v[1], ya.y, c); c = __dp4a(v[2], ya.z, c); c = __dp4a(v[3], ya.w, c);
        c = __dp4a(v[4], yb.x, c); c = __dp4a(v[5], yb.y, c); c = __dp4a(v[6], yb.z, c); c = __dp4a(v[7], yb.w, c);
        sc[wy][lane] = (float) c;
        sa[wy][lane] = __half2float(__ushort_as_half(dbits));
        sd[wy][lane] = dB;
        __syncwarp();
        for (int qq = 0; qq < 32; ++qq) {
            const int q = MODE == 4 ? 31 - qq : qq;
            const float Cf = sc[wy][q], dA = sa[wy][q], dq = sd[wy][q];
            if (MODE == 1) {
                s = __fmaf_rn(__fmul_rn(dA, dq), Cf, s);
            } else if (MODE == 2) {
                s = __fadd_rn(s, __fmul_rn(__fmul_rn(Cf, dA), dq));
            } else if (MODE == 3) {
                if (q & 1) s1 = __fmaf_rn(__fmul_rn(Cf, dA), dq, s1);
                else       s0 = __fmaf_rn(__fmul_rn(Cf, dA), dq, s0);
            } else {
                s = __fmaf_rn(__fmul_rn(Cf, dA), dq, s);
            }
        }
        __syncwarp();
    }
    if (MODE == 3) s = __fadd_rn(s0, s1);
    if (lane == 0) dst[row] = s;
}

// ---------------- input generators ----------------
struct rng_t {
    uint64_t s;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return (next() >> 11) * (1.0/9007199254740992.0); }
    float n() { double a = u(), b = u(); if (a < 1e-300) a = 1e-300; return (float) (std::sqrt(-2.0*std::log(a))*std::cos(6.283185307179586*b)); }
};

static const char * TNAME[] = {"normal", "subnormal", "zeros", "large", "cancel", "single"};
static const int NTYPES = 6;

static void gen(int type, uint64_t seed, int64_t K, int64_t M, std::vector<float> & w, std::vector<float> & x, int ncols) {
    rng_t r{seed*0x9E3779B97F4A7C15ull + 12345};
    for (int i = 0; i < 10; ++i) r.next();
    w.assign((size_t) K*M, 0.0f);
    x.assign((size_t) K*ncols, 0.0f);
    const int64_t nbk = K/32;
    switch (type) {
        case 0: // normal weights with per-row scale, normal activations with ~0.1% x20 outliers
            for (int64_t i = 0; i < M; ++i) { const float rs = 0.02f*std::exp(0.5f*r.n()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (auto & v : x) { v = r.n(); if ((r.next() & 1023) == 0) v *= 20.0f; }
            break;
        case 1: // tiny: weight rows with half-subnormal Q8_0 scales, activation blocks in/near float subnormal range
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, -8.0 + 5.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const float bs = (float) std::pow(10.0, -44.0 + 10.0*r.u());
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n();
            }
            break;
        case 2: // normal, with ~25% all-zero activation blocks (amax = 0) and ~25% all-zero weight blocks (d = 0)
            for (int64_t i = 0; i < M; ++i) for (int64_t b = 0; b < nbk; ++b) {
                const bool z = (r.next() & 3) == 0;
                for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = z ? 0.0f : 0.02f*r.n();
            }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const bool z = (r.next() & 3) == 0;
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = z ? 0.0f : r.n();
            }
            break;
        case 3: // large: |w| up to ~8e6 (Q8_0 d near half max), activations up to ~1e36 -> overflow to +-Inf in some rows
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, 2.0 + 4.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = std::max(-8.0e6f, std::min(8.0e6f, rs*r.n())); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const float bs = (float) std::pow(10.0, 20.0 + 16.0*r.u());
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n();
            }
            break;
        case 4: { // cancellation: w block b = (-1)^b * p_row, x block b = q * (1 + 1e-3 b) (+ tiny noise) -> alternating partial sums
            std::vector<float> q(32);
            for (int c = 0; c < ncols; ++c) {
                for (auto & v : q) v = r.n();
                for (int64_t b = 0; b < nbk; ++b) for (int k = 0; k < 32; ++k)
                    x[(size_t) c*K + b*32 + k] = q[k]*(1.0f + 1e-3f*b) + ((r.next() & 7) == 0 ? 1e-3f*r.n() : 0.0f);
            }
            std::vector<float> p(32);
            for (int64_t i = 0; i < M; ++i) {
                for (auto & v : p) v = 0.05f*r.n();
                const float mag = (float) std::pow(10.0, 3.0*r.u());
                for (int64_t b = 0; b < nbk; ++b) for (int k = 0; k < 32; ++k)
                    w[i*K + b*32 + k] = ((b & 1) ? -1.0f : 1.0f)*mag*p[k];
            }
        } break;
        case 5: // one nonzero element per activation block; half the weight rows also one nonzero per block
            for (int64_t i = 0; i < M; ++i) {
                const bool single = (i & 1);
                for (int64_t b = 0; b < nbk; ++b) {
                    const int pos = (int) (r.next() % 32);
                    for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = (!single || k == pos) ? 0.02f*r.n() : 0.0f;
                }
            }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const int pos = (int) (r.next() % 32);
                x[(size_t) c*K + b*32 + pos] = r.n()*(float) std::pow(10.0, -3.0 + 6.0*r.u());
            }
            break;
    }
}

typedef void (*set_bi_t)(ggml_backend_t, bool);

struct ggml_ref {
    ggml_backend_t be; set_bi_t set_bi; ggml_tensor * W; ggml_tensor * X; int64_t K, M;
    void run(int N, std::vector<float> & out) {
        set_bi(be, true);
        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * g = ggml_init(ip);
        ggml_tensor * xv = ggml_view_2d(g, X, K, N, X->nb[1], 0);
        ggml_tensor * o = ggml_mul_mat(g, W, xv);
        ggml_cgraph * gf = ggml_new_graph(g);
        ggml_build_forward_expand(gf, o);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
        ggml_backend_graph_compute(be, gf);
        out.resize((size_t) N*M);
        ggml_backend_tensor_get(o, out.data(), 0, ggml_nbytes(o));
        ggml_gallocr_free(ga);
        ggml_free(g);
    }
};

static inline uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s K M nseeds types(e.g. 012345) [label]\n", argv[0]); return 1; }
    const int64_t K = atoll(argv[1]), M = atoll(argv[2]);
    const int nseeds = atoi(argv[3]);
    const std::string types = argv[4];
    const char * label = argc > 5 ? argv[5] : "run";
    if (K % 1024 != 0) { fprintf(stderr, "K %% 1024 != 0\n"); return 1; }
    const int NSET[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};   // N=16 stays on forced MMQ in every build
    const int NCOL = 16;

    ggml_backend_t be = ggml_backend_cuda_init(0);
    set_bi_t set_bi = (set_bi_t) ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_set_batch_invariant");
    if (!be || !set_bi) { fprintf(stderr, "no backend\n"); return 1; }
    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOL);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_ref ref{be, set_bi, W, X, K, M};

    const int64_t nb = K/32;
    block_q8_1_mmq_t * yq; float * dout;
    CK(cudaMalloc(&yq, (K/128)*sizeof(block_q8_1_mmq_t)));
    CK(cudaMalloc(&dout, M*sizeof(float)));
    const dim3 qgrid(1, (unsigned)((K + 511)/512), 1), qblock(128);
    const char * Wd = (const char *) W->data; const int64_t rs = W->nb[1]; const int Mi = (int) M, nbi = (int) nb;

    // candidate ids: 0 prod, 1 ctrl-a assoc, 2 ctrl-b unfused, 3 ctrl-c splitK, 4 ctrl-d mmvq-quantizer, 5 ctrl-e reversed chunk order
    const char * CNAME[] = {"prod", "ctrl_a_assoc", "ctrl_b_unfused", "ctrl_c_splitK2", "ctrl_d_mmvq_quant", "ctrl_e_reverse"};
    const int NC = 6;
    auto launch = [&](int cand, const float * xj) {
        if (cand == 4) quant_mmvq_style<<<qgrid, qblock>>>(xj, yq, K, K);
        else           quant_mmq_d4<<<qgrid, qblock>>>(xj, yq, K, K);
        switch (cand) {
            case 0: case 4: gemv_prod<2><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, yq, dout, Mi, nbi); break;
            case 1: gemv_ctrl<1><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, yq, dout, Mi, nbi); break;
            case 2: gemv_ctrl<2><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, yq, dout, Mi, nbi); break;
            case 3: gemv_ctrl<3><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, yq, dout, Mi, nbi); break;
            case 5: gemv_ctrl<4><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, yq, dout, Mi, nbi); break;
        }
        CK(cudaGetLastError());
    };

    printf("label,K,M,type,seed,ref_selfinv_bad,cand,bad_vs_ref,compared,nonfinite_ref,nonfinite_cand,fp64_maxerr_norm,fp64_bad\n");
    std::vector<float> wf, xf, tmp;
    std::vector<uint8_t> wq((size_t) M*nb*34);
    std::vector<uint8_t> yh((K/128)*sizeof(block_q8_1_mmq_t));
    // summary
    std::vector<long long> tot_bad(NC, 0), tot_cmp(NC, 0), seeds_failing(NC, 0);
    long long tot_selfinv = 0, tot_fp64_bad = 0; double worst_fp64 = 0.0;
    for (char tc : types) {
        const int type = tc - '0';
        if (type < 0 || type >= NTYPES) continue;
        std::vector<long long> type_seeds_fail(NC, 0);
        for (int seed = 0; seed < nseeds; ++seed) {
            gen(type, (uint64_t) seed + 1000ull*type + 7ull*K + M, K, M, wf, xf, NCOL);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, M, K, nullptr);
            ggml_backend_tensor_set(W, wq.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, xf.data(), 0, ggml_nbytes(X));
            ggml_backend_synchronize(be);

            std::vector<std::vector<float>> R(17);
            for (int N : NSET) ref.run(N, R[N]);
            long long selfinv = 0, nonfin_ref = 0;
            for (int N : NSET) for (int j = 0; j < N; ++j) for (int64_t i = 0; i < M; ++i) {
                const float a = R[N][(size_t) j*M + i];
                if (fb(a) != fb(R[16][(size_t) j*M + i])) ++selfinv;
                if (!std::isfinite(a)) ++nonfin_ref;
            }
            tot_selfinv += selfinv;

            for (int cand = 0; cand < NC; ++cand) {
                long long bad = 0, cmp = 0, nonfin = 0, fp64bad = 0; double maxerr = 0.0;
                for (int j = 0; j < NCOL; ++j) {
                    const float * xj = (const float *) X->data + (size_t) j*K;
                    launch(cand, xj);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> mine(M);
                    CK(cudaMemcpy(mine.data(), dout, M*sizeof(float), cudaMemcpyDeviceToHost));
                    for (int N : NSET) {
                        if (j >= N) continue;
                        cmp += M;
                        if (memcmp(mine.data(), R[N].data() + (size_t) j*M, M*sizeof(float)) != 0) {
                            for (int64_t i = 0; i < M; ++i) if (fb(mine[i]) != fb(R[N][(size_t) j*M + i])) ++bad;
                        }
                    }
                    for (int64_t i = 0; i < M; ++i) if (!std::isfinite(mine[i])) ++nonfin;
                    // fp64 reference on the quantized operands, for the production kernel, columns 0..3
                    if (cand == 0 && j < 4) {
                        CK(cudaMemcpy(yh.data(), yq, yh.size(), cudaMemcpyDeviceToHost));
                        const block_q8_1_mmq_t * yb = (const block_q8_1_mmq_t *) yh.data();
                        for (int64_t i = 0; i < M; ++i) {
                            const block_q8_0_t * wb = (const block_q8_0_t *) (wq.data() + (size_t) i*nb*34);
                            double acc = 0.0, absacc = 0.0;
                            for (int64_t b = 0; b < nb; ++b) {
                                const block_q8_1_mmq_t & y1 = yb[b/4];
                                long long C = 0;
                                for (int k = 0; k < 32; ++k) C += (long long) wb[b].qs[k] * (long long) y1.qs[32*(b % 4) + k];
                                const double term = (double) C * (double) __half2float(wb[b].d) * (double) y1.d4[b % 4];
                                acc += term; absacc += std::fabs(term);
                            }
                            const double got = mine[i];
                            if (!std::isfinite(acc) || !std::isfinite(absacc)) continue;
                            if (std::fabs(acc) > 3.0e38) { if (std::isfinite(got) && std::fabs(got) < 3.0e38) { /* partial-sum overflow is legal */ } continue; }
                            if (!std::isfinite(got)) { if (absacc > 3.0e38) continue; ++fp64bad; continue; }
                            // bound: nb fma/mul roundings of 2^-24 each on |terms|, plus FTZ loss of up to 2*nb*FLT_MIN
                            const double tol = (double) (2*nb + 2) * std::ldexp(1.0, -24) * absacc + 2.0*nb*1.1754944e-38;
                            const double err = std::fabs(got - acc);
                            const double norm = absacc > 0 ? err / absacc : (err > 0 ? 1.0 : 0.0);
                            maxerr = std::max(maxerr, norm);
                            if (err > tol) ++fp64bad;
                        }
                    }
                }
                tot_bad[cand] += bad; tot_cmp[cand] += cmp;
                if (bad) { ++seeds_failing[cand]; ++type_seeds_fail[cand]; }
                if (cand == 0) { tot_fp64_bad += fp64bad; worst_fp64 = std::max(worst_fp64, maxerr); }
                printf("%s,%lld,%lld,%s,%d,%lld,%s,%lld,%lld,%lld,%lld,%.3e,%lld\n", label, (long long) K, (long long) M, TNAME[type], seed,
                       selfinv, CNAME[cand], bad, cmp, nonfin_ref, nonfin, cand == 0 ? maxerr : 0.0, cand == 0 ? fp64bad : 0LL);
            }
        }
        fprintf(stderr, "[%s K=%lld M=%lld %s] seeds failing per candidate:", label, (long long) K, (long long) M, TNAME[type]);
        for (int c = 0; c < NC; ++c) fprintf(stderr, " %s=%lld/%d", CNAME[c], type_seeds_fail[c], nseeds);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[%s K=%lld M=%lld] SUMMARY ref_selfinv_bad=%lld fp64_bad=%lld worst_fp64_norm=%.3e\n", label, (long long) K, (long long) M, tot_selfinv, tot_fp64_bad, worst_fp64);
    for (int c = 0; c < NC; ++c) fprintf(stderr, "  %-18s mismatching %lld / %lld floats, failing seeds %lld\n", CNAME[c], tot_bad[c], tot_cmp[c], seeds_failing[c]);
    cudaFree(yq); cudaFree(dout);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}
