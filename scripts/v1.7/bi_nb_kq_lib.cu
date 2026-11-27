// Library-level bit-identity check for the Q4_K / Q6_K N = 2..8 fast path (bi_nb_q4_K_v3 / bi_nb_q6_K_v3).
// For each seed and input family: run ggml_mul_mat in batch-invariant mode for N in {1..8, 16} on the same
// first-N columns, and compare every column bitwise against N = 16 (forced MMQ in every build).
// Negative control: the same N = 1..8 products in default mode (MMVQ) vs the BI N = 16 reference, which use
// different arithmetic and must mismatch. Also counts -0.0 and non-finite reference outputs (coverage).
// Usage: bi_nb_kq_lib q4_K|q6_K K M nseeds families(e.g. 012345) [label]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct rng_t {
    uint64_t s;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return (next() >> 11) * (1.0/9007199254740992.0); }
    float n() { double a = u(), b = u(); if (a < 1e-300) a = 1e-300; return (float) (std::sqrt(-2.0*std::log(a))*std::cos(6.283185307179586*b)); }
};

static const char * TNAME[] = {"normal", "subnormal", "zeros", "large", "cancel", "single", "negzero"};
static const int NTYPES = 7;

// families 0..5 as in bi_n1_ctrl.cu (32-value blocks); 6: activations mostly -0.0 with sparse tiny negatives
static void gen(int type, uint64_t seed, int64_t K, int64_t M, std::vector<float> & w, std::vector<float> & x, int ncols) {
    rng_t r{seed*0x9E3779B97F4A7C15ull + 12345};
    for (int i = 0; i < 10; ++i) r.next();
    w.assign((size_t) K*M, 0.0f);
    x.assign((size_t) K*ncols, 0.0f);
    const int64_t nbk = K/32;
    switch (type) {
        case 0:
            for (int64_t i = 0; i < M; ++i) { const float rs = 0.02f*std::exp(0.5f*r.n()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (auto & v : x) { v = r.n(); if ((r.next() & 1023) == 0) v *= 20.0f; }
            break;
        case 1:
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, -8.0 + 5.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = rs*r.n(); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const float bs = (float) std::pow(10.0, -44.0 + 10.0*r.u());
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n();
            }
            break;
        case 2:
            for (int64_t i = 0; i < M; ++i) for (int64_t b = 0; b < nbk; ++b) {
                const bool z = (r.next() & 3) == 0;
                for (int k = 0; k < 32; ++k) w[i*K + b*32 + k] = z ? 0.0f : 0.02f*r.n();
            }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const bool z = (r.next() & 3) == 0;
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = z ? 0.0f : r.n();
            }
            break;
        case 3: // large (K-quant scales are half: keep |w| below ~6e4*range so d stays finite)
            for (int64_t i = 0; i < M; ++i) { const float rs = (float) std::pow(10.0, 1.0 + 3.0*r.u()); for (int64_t k = 0; k < K; ++k) w[i*K + k] = std::max(-6.0e5f, std::min(6.0e5f, rs*r.n())); }
            for (int c = 0; c < ncols; ++c) for (int64_t b = 0; b < nbk; ++b) {
                const float bs = (float) std::pow(10.0, 20.0 + 16.0*r.u());
                for (int k = 0; k < 32; ++k) x[(size_t) c*K + b*32 + k] = bs*r.n();
            }
            break;
        case 4: {
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
        case 5:
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
        case 6:
            for (int64_t i = 0; i < M; ++i) for (int64_t k = 0; k < K; ++k) w[i*K + k] = (i & 1) ? -0.0f : 0.02f*r.n();
            for (int c = 0; c < ncols; ++c) for (int64_t k = 0; k < K; ++k)
                x[(size_t) c*K + k] = ((r.next() & 63) == 0) ? -1e-30f*std::fabs(r.n()) : -0.0f;
            break;
    }
}

typedef void (*set_bi_t)(ggml_backend_t, bool);

static void run(ggml_backend_t be, set_bi_t set_bi, bool bi, ggml_tensor * W, ggml_tensor * X, int64_t K, int64_t M, int N, std::vector<float> & out) {
    set_bi(be, bi);
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

static inline uint32_t fb(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s q4_K|q6_K K M nseeds families [label]\n", argv[0]); return 1; }
    const std::string tn = argv[1];
    const ggml_type wt = tn == "q4_K" ? GGML_TYPE_Q4_K : tn == "q6_K" ? GGML_TYPE_Q6_K : GGML_TYPE_COUNT;
    if (wt == GGML_TYPE_COUNT) { fprintf(stderr, "type\n"); return 1; }
    const int64_t K = atoll(argv[2]), M = atoll(argv[3]);
    const int nseeds = atoi(argv[4]);
    const std::string fams = argv[5];
    const char * label = argc > 6 ? argv[6] : "run";
    if (K % 512 != 0) { fprintf(stderr, "K %% 512 != 0 (fast-path gate)\n"); return 1; }
    const int NSET[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int NREF = 16;

    ggml_backend_t be = ggml_backend_cuda_init(0);
    set_bi_t set_bi = (set_bi_t) ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(), "ggml_backend_set_batch_invariant");
    if (!be || !set_bi) { fprintf(stderr, "no backend\n"); return 1; }
    ggml_init_params ip = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, K, M);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NREF);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    printf("label,type,K,M,family,seed,bi_bad,bi_cmp,ctrl_default_bad,ctrl_cmp,ref_negzero,ref_nonfinite\n");
    std::vector<float> wf, xf, R, O;
    std::vector<uint8_t> wq(ggml_row_size(wt, K)*M);
    long long T_bad = 0, T_cmp = 0, T_cbad = 0, T_ccmp = 0, T_nz = 0, T_nf = 0, seeds_bad = 0, ctrl_cells = 0, ctrl_fail_cells = 0;
    for (char fc : fams) {
        const int fam = fc - '0';
        if (fam < 0 || fam >= NTYPES) continue;
        for (int seed = 0; seed < nseeds; ++seed) {
            gen(fam, (uint64_t) seed + 1000ull*fam + 7ull*K + M + (wt == GGML_TYPE_Q6_K ? 77777ull : 0ull), K, M, wf, xf, NREF);
            ggml_quantize_chunk(wt, wf.data(), wq.data(), 0, M, K, nullptr);
            ggml_backend_tensor_set(W, wq.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(X, xf.data(), 0, ggml_nbytes(X));
            ggml_backend_synchronize(be);
            run(be, set_bi, true, W, X, K, M, NREF, R);
            long long nz = 0, nf = 0;
            for (float v : R) { if (fb(v) == 0x80000000u) ++nz; if (!std::isfinite(v)) ++nf; }
            long long bad = 0, cmp = 0, cbad = 0, ccmp = 0;
            for (int N : NSET) {
                run(be, set_bi, true, W, X, K, M, N, O);
                for (size_t i = 0; i < O.size(); ++i) { ++cmp; if (fb(O[i]) != fb(R[i])) ++bad; }
                run(be, set_bi, false, W, X, K, M, N, O);
                long long cb = 0;
                for (size_t i = 0; i < O.size(); ++i) { ++ccmp; if (fb(O[i]) != fb(R[i])) ++cb; }
                cbad += cb; ++ctrl_cells; if (cb > 0) ++ctrl_fail_cells;
            }
            T_bad += bad; T_cmp += cmp; T_cbad += cbad; T_ccmp += ccmp; T_nz += nz; T_nf += nf; if (bad) ++seeds_bad;
            printf("%s,%s,%lld,%lld,%s,%d,%lld,%lld,%lld,%lld,%lld,%lld\n", label, tn.c_str(), (long long) K, (long long) M, TNAME[fam], seed,
                   bad, cmp, cbad, ccmp, nz, nf);
        }
    }
    fprintf(stderr, "[%s %s K=%lld M=%lld] SUMMARY bi_bad=%lld/%lld seeds_bad=%lld ctrl_default_bad=%lld/%lld ctrl_cells_failing=%lld/%lld ref_negzero=%lld ref_nonfinite=%lld\n",
            label, tn.c_str(), (long long) K, (long long) M, T_bad, T_cmp, seeds_bad, T_cbad, T_ccmp, ctrl_fail_cells, ctrl_cells, T_nz, T_nf);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}
