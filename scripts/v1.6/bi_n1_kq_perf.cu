// K-quant fast N=1: v2 variant sweep. For each variant: exact-byte compare vs forced MMQ (ggml, BI on, N=1 and N=16
// columns) on 2 families x 3 seeds, then CUDA-event timing of quantize+gemv (5 rounds x 200 iterations, variant
// order rotated per round). Same harness and buffers for every variant.
// usage: bi_n1_kq_perf q4_K|q6_K K M
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "mmq-bi-n1-kq-v2.cuh"
#include "q8_prod_copy.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int LAYOUT>
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
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv); q.y = roundf(xi.y*d_inv); q.z = roundf(xi.z*d_inv); q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;
    float sum = 0.0f;
    if (LAYOUT == 1) {
        int sum_q = int(q.x) + int(q.y) + int(q.z) + int(q.w);
#pragma unroll
        for (int offset = 32/8; offset > 0; offset >>= 1) sum_q += __shfl_xor_sync(0xffffffffu, sum_q, offset, 32);
        sum = d * float(sum_q);
    }
    ((char4 *) (yb + 16))[iqs/4] = q;
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
typedef void (*set_bi_t)(ggml_backend_t, bool);

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 1; }
    const bool is_q4 = std::string(argv[1]) == "q4_K";
    const bool is_q8 = std::string(argv[1]) == "q8_0";
    const ggml_type gt = is_q8 ? GGML_TYPE_Q8_0 : is_q4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    const int64_t K = atoll(argv[2]), M = atoll(argv[3]);
    const int nsb = (int) (K/256), Mi = (int) M;
    ggml_backend_t be = ggml_backend_cuda_init(0);
    set_bi_t set_bi = (set_bi_t) ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_set_batch_invariant");
    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, gt, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 16);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    char * yq; float * dout;
    CK(cudaMalloc(&yq, (K/128)*144)); CK(cudaMalloc(&dout, M*sizeof(float)));
    const dim3 qgrid(1, (unsigned)((K + 511)/512), 1), qblock(128);
    const char * Wd = (const char *) W->data; const int64_t rs = W->nb[1];

    struct var { std::string name; std::function<void()> gemv; };
    std::vector<var> V;
#define V1(NWv) V.push_back({"v1_nw" #NWv, [&]{ if (is_q4) bi_n1_q4_K<NWv, 0><<<(Mi + NWv - 1)/NWv, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); \
                                             else bi_n1_q6_K<NWv, 0><<<(Mi + NWv - 1)/NWv, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); }})
#define V2(NWv, Rv, PFv) V.push_back({"v2_nw" #NWv "_r" #Rv "_pf" #PFv, [&]{ const int nb_ = (Mi + NWv*Rv - 1)/(NWv*Rv); \
        if (is_q4) bi_n1_q4_K_v2<NWv, Rv, PFv><<<nb_, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); \
        else       bi_n1_q6_K_v2<NWv, Rv, PFv><<<nb_, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); }})
    const int nb32 = (int) (K/32);
#define VQ8(NWv, RWv) V.push_back({"q8v3_nw" #NWv "_rw" #RWv, [&]{ const int nb_ = (Mi + NWv*RWv - 1)/(NWv*RWv); \
        bi_n1_q8_0_v3<NWv, RWv><<<nb_, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nb32); }})
    if (is_q8) {
        V.push_back({"q8_prod_v1", [&]{ gemv_prod<2><<<(Mi + 1)/2, dim3(32, 2)>>>(Wd, rs, (const block_q8_1_mmq_t *) yq, dout, Mi, nb32); }});
        VQ8(1, 4); VQ8(2, 4); VQ8(4, 4); VQ8(1, 8); VQ8(2, 8); VQ8(4, 8); VQ8(1, 16); VQ8(2, 16); VQ8(1, 32); VQ8(2, 2); VQ8(4, 2);
    } else {
    V1(2);
    V2(2, 1, false); V2(2, 1, true); V2(2, 2, false); V2(2, 2, true); V2(2, 4, false); V2(2, 4, true);
    V2(4, 1, false); V2(4, 1, true); V2(4, 2, false); V2(4, 2, true); V2(4, 4, false); V2(4, 4, true);
    V2(8, 2, true);
#define V3(NWv, RWv) V.push_back({"v3_nw" #NWv "_rw" #RWv, [&]{ const int nb_ = (Mi + NWv*RWv - 1)/(NWv*RWv); \
        if (is_q4) bi_n1_q4_K_v3<NWv, RWv><<<nb_, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); \
        else       bi_n1_q6_K_v3<NWv, RWv><<<nb_, dim3(32, NWv)>>>(Wd, rs, yq, dout, Mi, nsb); }})
    V3(2, 4); V3(4, 4); V3(1, 8); V3(2, 8); V3(4, 8); V3(1, 16); V3(2, 16); V3(1, 32);
    }
    auto quant = [&](const float * xj) { if (is_q4) quant_col<1><<<qgrid, qblock>>>(xj, yq, K, K); else quant_col<0><<<qgrid, qblock>>>(xj, yq, K, K); };

    // correctness: families normal (0) and mid (7), 3 seeds, all 16 columns vs ggml N=16 and column 0 vs N=1
    std::vector<long long> bad(V.size(), 0); long long cmp = 0;
    std::vector<float> wf((size_t) K*M), xf((size_t) K*16), r16((size_t) 16*M), r1(M), mine(M);
    std::vector<uint8_t> wq(ggml_nbytes(W));
    for (int fam = 0; fam < 2; ++fam) for (int seed = 0; seed < 3; ++seed) {
        rng_t r{(uint64_t) (seed + 31*fam + 1)*0x9E3779B97F4A7C15ull};
        for (int64_t i = 0; i < M; ++i) {
            const float s = fam == 0 ? 0.02f*std::exp(0.5f*r.n()) : (float) std::pow(10.0, -2.0 + 6.0*r.u());
            for (int64_t k = 0; k < K; ++k) wf[i*K + k] = fam == 0 ? s*r.n() : std::max(-3.0e4f, std::min(3.0e4f, s*r.n()));
        }
        for (int64_t b = 0; b < K*16/32; ++b) {
            const float s = fam == 0 ? 1.0f : (float) std::pow(10.0, -3.0 + 5.5*r.u());
            for (int k = 0; k < 32; ++k) xf[b*32 + k] = fam == 0 ? r.n() : std::max(-1500.0f, std::min(1500.0f, s*r.n()));
        }
        ggml_quantize_chunk(gt, wf.data(), wq.data(), 0, M, K, nullptr);
        ggml_backend_tensor_set(W, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(X, xf.data(), 0, ggml_nbytes(X));
        for (int N : {1, 16}) {
            set_bi(be, true);
            ggml_init_params gp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
            ggml_context * g = ggml_init(gp);
            ggml_tensor * o = ggml_mul_mat(g, W, ggml_view_2d(g, X, K, N, X->nb[1], 0));
            ggml_cgraph * gf = ggml_new_graph(g); ggml_build_forward_expand(gf, o);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(ga, gf); ggml_backend_graph_compute(be, gf);
            ggml_backend_tensor_get(o, N == 1 ? r1.data() : r16.data(), 0, ggml_nbytes(o));
            ggml_gallocr_free(ga); ggml_free(g);
        }
        for (size_t v = 0; v < V.size(); ++v) {
            for (int j = 0; j < 16; ++j) {
                quant((const float *) X->data + (size_t) j*K);
                CK(cudaMemset(dout, 0xff, M*sizeof(float)));
                V[v].gemv(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(mine.data(), dout, M*sizeof(float), cudaMemcpyDeviceToHost));
                if (memcmp(mine.data(), r16.data() + (size_t) j*M, M*4)) for (int64_t i = 0; i < M; ++i) bad[v] += memcmp(&mine[i], &r16[(size_t) j*M + i], 4) != 0;
                if (j == 0 && memcmp(mine.data(), r1.data(), M*4)) for (int64_t i = 0; i < M; ++i) bad[v] += memcmp(&mine[i], &r1[i], 4) != 0;
                if (v == 0) cmp += M + (j == 0 ? M : 0);
            }
        }
    }
    // timing
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const float * x0 = (const float *) X->data;
    const int ITERS = 200, ROUNDS = 5;
    std::vector<std::vector<double>> t(V.size());
    for (size_t v = 0; v < V.size(); ++v) for (int w = 0; w < 50; ++w) { quant(x0); V[v].gemv(); }
    CK(cudaDeviceSynchronize());
    for (int round = 0; round < ROUNDS; ++round) {
        for (size_t vi = 0; vi < V.size(); ++vi) {
            const size_t v = (vi + round*5) % V.size();
            CK(cudaEventRecord(e0));
            for (int it = 0; it < ITERS; ++it) { quant(x0); V[v].gemv(); }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            t[v].push_back(1e3*ms/ITERS);
        }
    }
    printf("type,K,M,variant,bad,compared,us_med,us_min,us_max,ratio_vs_v1\n");
    auto med = [](std::vector<double> x) { std::sort(x.begin(), x.end()); return x[x.size()/2]; };
    const double base = med(t[0]);
    for (size_t v = 0; v < V.size(); ++v) {
        printf("%s,%lld,%lld,%s,%lld,%lld,%.2f,%.2f,%.2f,%.3f\n", argv[1], (long long) K, (long long) M, V[v].name.c_str(), bad[v], cmp,
               med(t[v]), *std::min_element(t[v].begin(), t[v].end()), *std::max_element(t[v].begin(), t[v].end()), med(t[v])/base);
    }
    cudaFree(yq); cudaFree(dout); ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}
