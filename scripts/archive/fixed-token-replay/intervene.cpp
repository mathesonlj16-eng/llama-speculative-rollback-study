// Experiment E5: causal interventions at layer 15, CPU, one thread, n_rs_seq=0.
// A = chunks 7+1+3, B = chunks 7+4. Same tokens and positions.
//   control      : A with a no-op callback. Must be bit-identical to A.
//   patch-rope   : B, with the final-token row of Qcur-15 (ROPE) replaced by A's bits.
//   patch-attn   : B, with the final-token row of layer 15 FLASH_ATTN_EXT replaced by A's bits.
//   ulp-rope     : A, with each final-token Qcur-15 element moved by one ULP in a random direction.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const std::vector<int> TOKENS = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};
enum class mode { record, none, patch_rope, patch_attn, ulp_rope };
struct state {
    mode m = mode::none;
    bool last_decode = false;
    int n_tokens = 0, flash_seen = 0, applied = 0;
    unsigned seed = 0;
    std::vector<float> rope, attn;
};

static void slice(ggml_tensor *t, int n_tokens, size_t &offset, size_t &count) {
    if (t->ne[2] != n_tokens || t->ne[3] != 1) throw std::runtime_error("unexpected shape");
    count = t->ne[0] * t->ne[1];
    offset = (n_tokens - 1) * count * sizeof(float);
    if (t->nb[1] != t->ne[0] * (int64_t) sizeof(float) || t->nb[2] != (int64_t) count * (int64_t) sizeof(float))
        throw std::runtime_error("non-contiguous tensor");
}

static bool on_eval(ggml_tensor *t, bool ask, void *u) {
    auto *s = static_cast<state *>(u);
    const bool rope = t->op == GGML_OP_ROPE && !strcmp(ggml_get_name(t), "Qcur-15");
    const bool flash = t->op == GGML_OP_FLASH_ATTN_EXT;
    if (ask) return rope || flash;
    if (!s->last_decode || t->type != GGML_TYPE_F32) return true;
    bool attn15 = false;
    if (flash) attn15 = ++s->flash_seen == 4;  // full-attention layers are 3, 7, 11, 15
    if (!rope && !attn15) return true;
    size_t off, n;
    slice(t, s->n_tokens, off, n);
    if (s->m == mode::record) {
        auto &dst = rope ? s->rope : s->attn;
        dst.resize(n);
        ggml_backend_tensor_get(t, dst.data(), off, n * sizeof(float));
    } else if ((s->m == mode::patch_rope && rope) || (s->m == mode::patch_attn && attn15)) {
        const auto &src = rope ? s->rope : s->attn;
        ggml_backend_tensor_set(t, src.data(), off, n * sizeof(float));
        s->applied++;
    } else if (s->m == mode::ulp_rope && rope) {
        std::vector<float> v(n);
        ggml_backend_tensor_get(t, v.data(), off, n * sizeof(float));
        std::mt19937 rng(s->seed);
        for (auto &x : v) x = std::nextafter(x, (rng() & 1) ? INFINITY : -INFINITY);
        ggml_backend_tensor_set(t, v.data(), off, n * sizeof(float));
        s->applied++;
    }
    return true;
}

static std::vector<float> run(llama_model *m, const std::vector<int> &sizes, state &st) {
    st.flash_seen = 0; st.applied = 0;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 1;
    cp.n_seq_max = 2; cp.n_rs_seq = 0; cp.kv_unified = false; cp.cb_eval = on_eval; cp.cb_eval_user_data = &st;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    std::vector<float> out;
    int pos = 0;
    for (size_t k = 0; k < sizes.size(); k++) {
        st.last_decode = k + 1 == sizes.size();
        st.n_tokens = sizes[k];
        st.flash_seen = 0;
        auto *b = llama_batch_ext_init(c);
        for (int j = 0; j < sizes[k]; j++) {
            const int q = pos + j;
            const bool last = q + 1 == (int) TOKENS.size();
            int i = llama_batch_ext_add_token(b, 1, TOKENS[q]);
            if (i < 0 || !llama_batch_ext_set_pos(b, i, &q) || !llama_batch_ext_set_output_logits(b, i, last))
                throw std::runtime_error("batch");
        }
        if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
        pos += sizes[k];
        if (st.last_decode) {
            auto *z = llama_get_logits_ith(c, sizes[k] - 1);
            out.assign(z, z + llama_vocab_n_tokens(llama_model_get_vocab(m)));
        }
        llama_batch_ext_free(b);
    }
    llama_free(c);
    return out;
}

static void report(const char *name, int seed, const std::vector<float> &z, const std::vector<float> &a,
                   const std::vector<float> &b, int applied) {
    double da = 0, db = 0;
    for (size_t i = 0; i < z.size(); i++) {
        da = std::max<double>(da, std::fabs(z[i] - a[i]));
        db = std::max<double>(db, std::fabs(z[i] - b[i]));
    }
    printf("%s,%d,%d,%td,%.6f,%.6f,%.6f,%d,%d\n", name, seed, applied, std::max_element(z.begin(), z.end()) - z.begin(),
           z[27355] - z[11767], da, db, !memcmp(z.data(), a.data(), z.size() * sizeof(float)),
           !memcmp(z.data(), b.data(), z.size() * sizeof(float)));
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    const std::vector<int> A{7, 1, 3}, B{7, 4};
    state st;
    st.m = mode::record;
    auto la = run(m, A, st);
    st.m = mode::none;
    auto lb = run(m, B, st);
    printf("case,seed,applied,top,margin_sorting_minus_algorithm,max_abs_diff_vs_A,max_abs_diff_vs_B,bit_equal_A,bit_equal_B\n");
    report("baseline-A", 0, la, la, lb, 0);
    report("baseline-B", 0, lb, la, lb, 0);
    st.m = mode::none;
    auto z = run(m, A, st); report("control-A", 0, z, la, lb, st.applied);
    st.m = mode::patch_rope;
    z = run(m, B, st); report("patch-rope", 0, z, la, lb, st.applied);
    st.m = mode::patch_attn;
    z = run(m, B, st); report("patch-attn", 0, z, la, lb, st.applied);
    st.m = mode::ulp_rope;
    for (unsigned seed = 1; seed <= 32; seed++) {
        st.seed = seed;
        z = run(m, A, st);
        report("ulp-rope", seed, z, la, lb, st.applied);
    }
    llama_model_free(m);
    llama_backend_free();
}
