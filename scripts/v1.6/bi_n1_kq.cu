// K-quant (Q4_K, Q6_K) fast N=1 kernels vs PR #32081 forced MMQ: exact bytes, negative controls,
// adversarial inputs, fp64 reference. Kernels come from the in-tree header (same source as the build):
//   worktrees/mmq-n1-fa/ggml/src/ggml-cuda/mmq-bi-n1-kq.cuh
// Reference = ggml MUL_MAT, batch-invariant on, N in {1,2,3,5,8,16}. Linked against build/pr-bi every N is
// forced MMQ; linked against build/mmq-n1-fa (after wiring), N=1 is the in-tree fast path.
// Build: nvcc -O3 -use_fast_math -arch=sm_75 -I<ggml-cuda dir> -I<ggml include>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "mmq-bi-n1-kq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// copy of quantize_mmq_q8_1<layout> math (quantize.cu:456-570 @ 3f25ea199), one column. LAYOUT 0 = D4, 1 = DS4.
// MMVQ = true: control (d): MMVQ-style math (d = amax/127, q = round(x/d), half d).
template <int LAYOUT, bool MMVQ>
__global__ void quant_col(const float * __restrict__ x, char * __restrict__ vy, const int64_t ne00, const int64_t ne0) {
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) return;
    const float4 * x4 = (const float4 *) x;
    const int64_t k_block = i0 / 128, iqs = i0 % 128;
    char * yb = vy + k_block*144;
    const float4 xi = i0 < ne00 ? x4[i0/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y)); amax = fmaxf(amax, fabsf(xi.z)); amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
    char4 q; float d;
    if (MMVQ) {
        d = amax / 127.0f;
        q.x = amax == 0.0f ? 0 : roundf(xi.x / d); q.y = amax == 0.0f ? 0 : roundf(xi.y / d);
        q.z = amax == 0.0f ? 0 : roundf(xi.z / d); q.w = amax == 0.0f ? 0 : roundf(xi.w / d);
        d = __half2float(__float2half(d));
    } else {
        const float d_inv = 127.0f / amax;
        q.x = roundf(xi.x*d_inv); q.y = roundf(xi.y*d_inv); q.z = roundf(xi.z*d_inv); q.w = roundf(xi.w*d_inv);
        d = 1.0f / d_inv;
    }
    float sum = 0.0f;
    if (LAYOUT == 1) {
        int sum_q = int(q.x) + int(q.y) + int(q.z) + int(q.w);
#pragma unroll
        for (int offset = 32/8; offset > 0; offset >>= 1) sum_q += __shfl_xor_sync(0xffffffffu, sum_q, offset, 32);
        sum = d * float(sum_q);
    }
    char4 * yqs4 = (char4 *) (yb + 16);
    yqs4[iqs/4] = q;
    if (iqs % 32 == 0) {
        if (LAYOUT == 1) ((half2 *) yb)[iqs/32] = make_half2(d, sum);
        else             ((float *) yb)[iqs/32] = d;
    }
}

struct rng_t {
    uint64_t s;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return (next() >> 11) * (1.0/9007199254740992.0); }
    float n() { double a = u(), b = u(); if (a < 1e-300) a = 1e-300; return (float) (std::sqrt(-2.0*std::log(a))*std::cos(6.283185307179586*b)); }
};
static const char * TNAME[] = {"normal", "subnormal", "zeros", "large", "cancel", "single", "large16", "mid"};

// same families as bi_n1_ctrl.cu; 'large' weights capped so that the K-quant fp16 super-scales stay finite
static void gen(int type, uint64_t seed, int64_t K, int64_t M, std::vector<float> & w, std::vector<float> & x, int ncols) {
    rng_t r{seed*0x9E3779B97F4A7C15ull + 12345};
    for (int i = 0; i < 10; ++i) r.next();
    w.assign((size_t) K*M, 0.0f); x.assign((size_t) K*ncols, 0.0f);
    const int64_t nbk = K/32;
    switch (type) {
        case 0:
            for (int64_t i = 0; i < M; ++i) { const float rs = 0.02f*std::exp(0.5f*r.n()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (auto & v : x) { v = r.n(); if ((r.next() & 1023) == 0) v *= 20.0f; }
            break;
        case 1:
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, -8.0 + 5.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const float bs = (float) std::pow(10.0, -44.0 + 10.0*r.u()); for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n(); }
            break;
        case 2:
            for (int64_t i = 0; i < M; ++i) for (int64_t b = 0; b < nbk; ++b) { const bool z = (r.next() & 3) == 0; for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = z ? 0.0f : 0.02f*r.n(); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const bool z = (r.next() & 3) == 0; for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = z ? 0.0f : r.n(); }
            break;
        case 3:
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, 1.0 + 2.5*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = std::max(-3.0e4f, std::min(3.0e4f, rs*r.n())); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const float bs = (float) std::pow(10.0, 20.0 + 16.0*r.u()); for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n(); }
            break;
        case 4: {
            std::vector<float> q(32), p(32);
            for (int c = 0; c < ncols; ++c) { for (auto & v : q) v = r.n(); for (int64_t b = 0; b < nbk; ++b) for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = q[k]*(1.0f + 1e-3f*b) + ((r.next() & 7) == 0 ? 1e-3f*r.n() : 0.0f); }
            for (int64_t i = 0; i < M; ++i) { for (auto & v : p) v = 0.05f*r.n(); const float mag = (float) std::pow(10.0, 3.0*r.u());
                for (int64_t b = 0; b < nbk; ++b) for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = ((b & 1) ? -1.0f : 1.0f)*mag*p[k]; }
        } break;
        case 6: // large but within the fp16 range of the K-quant/DS4 scales: |w| up to 3e4, |x| up to ~5e6
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, 1.0 + 3.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = std::max(-3.0e4f, std::min(3.0e4f, rs*r.n())); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const float bs = (float) std::pow(10.0, 3.0 + 3.0*r.u()); for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = std::max(-5.0e6f, std::min(5.0e6f, bs*r.n())); }
            break;
        case 7: // large finite: DS4 keeps s = d*sum(q) in fp16, so per-block |x| <= 1500 (s <= ~4.8e4); wide range of row/block scales
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, -2.0 + 6.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = std::max(-3.0e4f, std::min(3.0e4f, rs*r.n())); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const float bs = (float) std::pow(10.0, -3.0 + 5.5*r.u()); for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = std::max(-1500.0f, std::min(1500.0f, bs*r.n())); }
            break;
        case 5:
            for (int64_t i = 0; i < M; ++i) { const bool single = (i & 1);
                for (int64_t b = 0; b < nbk; ++b) { const int pos = (int) (r.next() % 32); for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = (!single || k == pos) ? 0.02f*r.n() : 0.0f; } }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) { const int pos = (int) (r.next() % 32); x[(size_t) c*K + b*32 + pos] = r.n()*(float) std::pow(10.0, -3.0 + 6.0*r.u()); }
            break;
    }
}

typedef void (*set_bi_t)(ggml_backend_t, bool);
static inline uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void ref_run(ggml_backend_t be, set_bi_t set_bi, ggml_tensor * W, ggml_tensor * X, int64_t K, int64_t M, int N, std::vector<float> & out) {
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
    ggml_gallocr_free(ga); ggml_free(g);
}

// fp64 reference of the defined program's exact value, from quantized bytes (host)
static double fp64_q4_K(const uint8_t * wrow, const uint8_t * y, int nsb, double & absacc) {
    double acc = 0.0; absacc = 0.0;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t * b = wrow + 144*sb;
        half d, dmin; memcpy(&d, b, 2); memcpy(&dmin, b + 2, 2);
        int scw[3]; memcpy(scw, b + 4, 12);
        for (int s = 0; s < 8; ++s) {
            auto unpack = [&](int ksc) { return ((scw[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | ((scw[ksc/2] >> (2 * (ksc % 2))) & 0x30303030); };
            const uint8_t sc = (uint8_t) (unpack(s/4) >> (8*(s%4))), m = (uint8_t) (unpack(s/4 + 2) >> (8*(s%4)));
            const float dax = __half2float(__float2half_rn(__half2float(d)*(float) sc));     // exact product, one fp16 rounding
            const float day = __half2float(__float2half_rn(-__half2float(dmin)*(float) m));
            const int g = 8*sb + s;
            const uint8_t * yb = y + (size_t) (g/4)*144;
            half2 ds; memcpy(&ds, yb + 4*(g%4), 4);
            const float dsx = __low2float(ds), dsy = __high2float(ds);
            long long C = 0;
            for (int k = 0; k < 32; ++k) {
                const int qv = (b[16 + 32*(s/2) + k] >> (4*(s & 1))) & 0xF;
                C += (long long) qv * (int8_t) yb[16 + 32*(g%4) + k];
            }
            const double t1 = (double) dax*(double) dsx*(double) C, t2 = (double) day*(double) dsy;
            acc += t1 + t2; absacc += std::fabs(t1) + std::fabs(t2);
        }
    }
    return acc;
}
static double g_q6_maxtmp = 0.0;   // max over half superblocks of sum |Cf*dB| (bound on |tmp| in the fp32 program)
static double fp64_q6_K(const uint8_t * wrow, const uint8_t * y, int nsb, double & absacc) {
    double acc = 0.0; absacc = 0.0; g_q6_maxtmp = 0.0;
    double tmpabs = 0.0;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t * b = wrow + 210*sb;
        half d; memcpy(&d, b + 208, 2);
        const double dA = __half2float(d);
        for (int q = 0; q < 8; ++q) {
            if (q % 4 == 0) tmpabs = 0.0;
            const int n = q/4, j = q%4;
            const int g = 8*sb + q;
            const uint8_t * yb = y + (size_t) (g/4)*144;
            float dB; memcpy(&dB, yb + 4*(g%4), 4);
            long long C[2] = {0, 0};
            for (int l = 0; l < 32; ++l) {
                const int ql = (b[64*n + 32*(j & 1) + l] >> (4*(j >> 1))) & 0xF;
                const int qh = (b[128 + 32*n + l] >> (2*j)) & 3;
                const int qv = (ql | (qh << 4)) - 32;
                C[l/16] += (long long) qv * (int8_t) yb[16 + 32*(g%4) + l];
            }
            const int8_t * sc = (const int8_t *) (b + 192);
            const double ct = (double) (C[0]*sc[2*q] + C[1]*sc[2*q + 1]) * (double) dB;
            tmpabs += std::fabs(ct);
            g_q6_maxtmp = std::max(g_q6_maxtmp, tmpabs);
            const double t = ct * dA;
            acc += t; absacc += std::fabs(t);
        }
    }
    return acc;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s q4_K|q6_K K M nseeds types [label]\n", argv[0]); return 1; }
    const bool is_q4 = std::string(argv[1]) == "q4_K";
    const ggml_type gt = is_q4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    const int64_t K = atoll(argv[2]), M = atoll(argv[3]);
    const int nseeds = atoi(argv[4]);
    const std::string types = argv[5];
    const char * label = argc > 6 ? argv[6] : "run";
    const int nsb = (int) (K/256);
    if (K % 256 != 0 || (!is_q4 && nsb % 2 != 0) || K % 512 != 0) { fprintf(stderr, "unsupported K\n"); return 1; }
    const int NSET[] = {1, 2, 3, 5, 8, 16};
    const int NCOL = 16;
    const int bpsb = is_q4 ? 144 : 210;

    ggml_backend_t be = ggml_backend_cuda_init(0);
    set_bi_t set_bi = (set_bi_t) ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_set_batch_invariant");
    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, gt, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOL);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if ((size_t) W->nb[1] != (size_t) bpsb*nsb) { fprintf(stderr, "row stride mismatch\n"); return 1; }

    char * yq; float * dout;
    CK(cudaMalloc(&yq, (K/128)*144));
    CK(cudaMalloc(&dout, M*sizeof(float)));
    const dim3 qgrid(1, (unsigned)((K + 511)/512), 1), qblock(128);
    const char * Wd = (const char *) W->data; const int64_t rs = W->nb[1]; const int Mi = (int) M;

    const char * CN4[] = {"prod", "ctrl_assoc", "ctrl_unfused_m", "ctrl_splitK2", "ctrl_fp32_dmA", "ctrl_m_first", "ctrl_mmvq_quant"};
    const char * CN6[] = {"prod", "ctrl_flat_fold", "ctrl_tmp_fmul", "ctrl_splitK2", "ctrl_fp32_scale", "ctrl_reverse", "ctrl_mmvq_quant"};
    const char ** CN = is_q4 ? CN4 : CN6;
    const int NC = 7;
    auto launch = [&](int cand, const float * xj) {
        if (is_q4) { if (cand == 6) quant_col<1, true><<<qgrid, qblock>>>(xj, yq, K, K); else quant_col<1, false><<<qgrid, qblock>>>(xj, yq, K, K); }
        else       { if (cand == 6) quant_col<0, true><<<qgrid, qblock>>>(xj, yq, K, K); else quant_col<0, false><<<qgrid, qblock>>>(xj, yq, K, K); }
        const dim3 gg((Mi + 1)/2), gb(32, 2);
        if (is_q4) {
            switch (cand) {
                case 0: case 6: bi_n1_q4_K<2, 0><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 1: bi_n1_q4_K<2, 1><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 2: bi_n1_q4_K<2, 2><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 3: bi_n1_q4_K<2, 3><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 4: bi_n1_q4_K<2, 4><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 5: bi_n1_q4_K<2, 5><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
            }
        } else {
            switch (cand) {
                case 0: case 6: bi_n1_q6_K<2, 0><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 1: bi_n1_q6_K<2, 1><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 2: bi_n1_q6_K<2, 2><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 3: bi_n1_q6_K<2, 3><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 4: bi_n1_q6_K<2, 4><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
                case 5: bi_n1_q6_K<2, 5><<<gg, gb>>>(Wd, rs, yq, dout, Mi, nsb); break;
            }
        }
        CK(cudaGetLastError());
    };

    printf("label,qtype,K,M,family,seed,ref_selfinv_bad,cand,bad_vs_ref,compared,nonfinite_cand,fp64_maxerr_norm,fp64_bad\n");
    std::vector<float> wf, xf;
    std::vector<uint8_t> wq((size_t) M*nsb*bpsb), yh((K/128)*144);
    std::vector<long long> tot_bad(NC, 0), tot_cmp(NC, 0), fail_seeds(NC, 0);
    long long tot_selfinv = 0, tot_fp64_bad = 0, cells = 0, interm_ovf = 0; double worst = 0.0;
    for (char tc : types) {
        const int fam = tc - '0';
        if (fam < 0 || fam > 7) continue;
        std::vector<long long> fam_fail(NC, 0);
        for (int seed = 0; seed < nseeds; ++seed) {
            gen(fam, (uint64_t) seed + 1000ull*fam + 7ull*K + M + (is_q4 ? 0 : 77777), K, M, wf, xf, NCOL);
            ggml_quantize_chunk(gt, wf.data(), wq.data(), 0, M, K, nullptr);
            ggml_backend_tensor_set(W, wq.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, xf.data(), 0, ggml_nbytes(X));
            ggml_backend_synchronize(be);
            std::vector<std::vector<float>> R(17);
            for (int N : NSET) ref_run(be, set_bi, W, X, K, M, N, R[N]);
            long long selfinv = 0;
            for (int N : NSET) for (int j = 0; j < N; ++j) for (int64_t i = 0; i < M; ++i)
                if (fb(R[N][(size_t) j*M + i]) != fb(R[16][(size_t) j*M + i])) ++selfinv;
            tot_selfinv += selfinv; ++cells;
            for (int cand = 0; cand < NC; ++cand) {
                long long bad = 0, cmp = 0, nonfin = 0, fbad = 0; double maxerr = 0.0;
                for (int j = 0; j < NCOL; ++j) {
                    launch(cand, (const float *) X->data + (size_t) j*K);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> mine(M);
                    CK(cudaMemcpy(mine.data(), dout, M*sizeof(float), cudaMemcpyDeviceToHost));
                    for (int N : NSET) {
                        if (j >= N) continue;
                        cmp += M;
                        if (memcmp(mine.data(), R[N].data() + (size_t) j*M, M*sizeof(float)) != 0)
                            for (int64_t i = 0; i < M; ++i) if (fb(mine[i]) != fb(R[N][(size_t) j*M + i])) ++bad;
                    }
                    for (int64_t i = 0; i < M; ++i) if (!std::isfinite(mine[i])) ++nonfin;
                    if (cand == 0 && j < 2) {
                        CK(cudaMemcpy(yh.data(), yq, yh.size(), cudaMemcpyDeviceToHost));
                        const int64_t rows_chk = std::min<int64_t>(M, 4096);
                        for (int64_t i = 0; i < rows_chk; ++i) {
                            double absacc;
                            const double ex = is_q4 ? fp64_q4_K(wq.data() + (size_t) i*nsb*bpsb, yh.data(), nsb, absacc)
                                                    : fp64_q6_K(wq.data() + (size_t) i*nsb*bpsb, yh.data(), nsb, absacc);
                            const double got = mine[i];
                            if (!std::isfinite(ex) || !std::isfinite(absacc) || std::fabs(ex) > 3.0e38 || absacc > 3.0e38) continue;
                            if (!is_q4 && g_q6_maxtmp > 3.0e38) { ++interm_ovf; continue; }   // legal: fp32 tmp overflows before *d
                            if (!std::isfinite(got)) { ++fbad; continue; }
                            const double tol = (double) (4*8*nsb + 4) * std::ldexp(1.0, -24) * absacc + 4.0*8*nsb*1.1754944e-38;
                            const double err = std::fabs(got - ex);
                            maxerr = std::max(maxerr, absacc > 0 ? err/absacc : (err > 0 ? 1.0 : 0.0));
                            if (err > tol) ++fbad;
                        }
                    }
                }
                tot_bad[cand] += bad; tot_cmp[cand] += cmp;
                if (bad) { ++fail_seeds[cand]; ++fam_fail[cand]; }
                if (cand == 0) { tot_fp64_bad += fbad; worst = std::max(worst, maxerr); }
                printf("%s,%s,%lld,%lld,%s,%d,%lld,%s,%lld,%lld,%lld,%.3e,%lld\n", label, argv[1], (long long) K, (long long) M, TNAME[fam], seed,
                       selfinv, CN[cand], bad, cmp, nonfin, cand == 0 ? maxerr : 0.0, cand == 0 ? fbad : 0LL);
            }
        }
        fprintf(stderr, "[%s %s K=%lld M=%lld %s] seeds failing:", label, argv[1], (long long) K, (long long) M, TNAME[fam]);
        for (int c = 0; c < NC; ++c) fprintf(stderr, " %s=%lld/%d", CN[c], fam_fail[c], nseeds);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[%s %s K=%lld M=%lld] SUMMARY cells=%lld ref_selfinv_bad=%lld fp64_bad=%lld fp64_skipped_intermediate_overflow=%lld worst_fp64_norm=%.3e\n", label, argv[1], (long long) K, (long long) M, cells, tot_selfinv, tot_fp64_bad, interm_ovf, worst);
    for (int c = 0; c < NC; ++c) fprintf(stderr, "  %-16s mismatching %lld / %lld floats, failing seeds %lld\n", CN[c], tot_bad[c], tot_cmp[c], fail_seeds[c]);
    cudaFree(yq); cudaFree(dout); ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}
