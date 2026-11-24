// Experiment E9: separates rollback leakage from chunking in the E8 draft-model mismatch.
//   X       : [anchor, a accepted, r rejected tokens W], roll back r, finish one token at a time.
//   canary  : X with different rejected tokens. Bit-equality means rejected token content left no trace.
//   truedraft: X with the true continuation in the rejected positions, still rolled back.
//   chunk   : the same first chunk using true tokens with no rollback, then finish one token at a time.
//   ref     : every token decoded alone.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using vec = std::vector<float>;
static const std::vector<int> T = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760, 1287, 314, 220};
static const std::vector<int> W = {9927, 279, 1287};
static const std::vector<int> C = {5000, 7000, 9000};
static constexpr int START = 7, LAST = 10, SEQ = 1;
struct item { int pos, id; };

static vec decode(llama_context *c, const std::vector<item> &xs, bool want) {
    auto *b = llama_batch_ext_init(c);
    for (size_t k = 0; k < xs.size(); k++) {
        const bool out = want && k + 1 == xs.size();
        int i = llama_batch_ext_add_token(b, SEQ, xs[k].id);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &xs[k].pos) || !llama_batch_ext_set_output_logits(b, i, out))
            throw std::runtime_error("batch");
    }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    vec out;
    if (want) {
        auto *z = llama_get_logits_ith(c, xs.size() - 1);
        out.assign(z, z + llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c))));
    }
    llama_batch_ext_free(b);
    return out;
}
static llama_context *make(llama_model *m) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 8;
    cp.n_seq_max = 2; cp.n_rs_seq = 3; cp.kv_unified = false;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    for (int i = 0; i < START; i++) decode(c, {{i, T[i]}}, false);
    return c;
}
static vec finish(llama_context *c, int from) {
    vec z;
    for (int i = from; i <= LAST; i++) z = decode(c, {{i, T[i]}}, i == LAST);
    return z;
}
static vec rollback(llama_model *m, int a, int r, const std::vector<int> &rejected) {
    auto *c = make(m);
    std::vector<item> xs{{START, T[START]}};
    for (int i = 0; i < a; i++) xs.push_back({START + 1 + i, T[START + 1 + i]});
    for (int i = 0; i < r; i++) xs.push_back({START + 1 + a + i, rejected[i]});
    decode(c, xs, false);
    if (!llama_memory_seq_rm(llama_get_memory(c), SEQ, START + 1 + a, -1)) throw std::runtime_error("rm");
    auto z = finish(c, START + 1 + a);
    llama_free(c);
    return z;
}
static vec chunk_only(llama_model *m, int n) {
    auto *c = make(m);
    std::vector<item> xs;
    for (int i = 0; i < n; i++) xs.push_back({START + i, T[START + i]});
    vec z = decode(c, xs, START + n - 1 == LAST);
    if (START + n <= LAST) z = finish(c, START + n);
    llama_free(c);
    return z;
}
static bool same(const vec &a, const vec &b) { return !memcmp(a.data(), b.data(), a.size() * sizeof(float)); }
static double diff(const vec &a, const vec &b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) d = std::max<double>(d, std::fabs(a[i] - b[i]));
    return d;
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    auto *c = make(m);
    const vec ref = finish(c, START);
    llama_free(c);
    printf("backend,scenario,x_eq_ref,x_eq_canary,x_eq_truedraft,x_eq_chunk,x_diff_ref,canary_diff,truedraft_diff,chunk_diff_ref,x_top,ref_top\n");
    for (int d = 1; d <= 3; d++) for (int a = 0; a < d; a++) {
        const int r = d - a;
        std::vector<int> truth;
        for (int i = 0; i < r; i++) truth.push_back(T[START + 1 + a + i]);
        auto x = rollback(m, a, r, W), can = rollback(m, a, r, C), td = rollback(m, a, r, truth), ch = chunk_only(m, d + 1);
        printf("%s,d%d-a%d-r%d,%d,%d,%d,%d,%.8g,%.8g,%.8g,%.8g,%td,%td\n", argv[3], d, a, r, same(x, ref), same(x, can), same(x, td), same(x, ch),
               diff(x, ref), diff(x, can), diff(x, td), diff(ch, ref), std::max_element(x.begin(), x.end()) - x.begin(),
               std::max_element(ref.begin(), ref.end()) - ref.begin());
        fflush(stdout);
    }
    llama_model_free(m);
    llama_backend_free();
}
