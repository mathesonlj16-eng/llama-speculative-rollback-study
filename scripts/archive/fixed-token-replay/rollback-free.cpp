// Experiment E7: checkpoint-free rollback feasibility at library level.
// Uses n_rs_seq=3 and llama_memory_seq_rm only. Every decode has at most 3 tokens, the chunk-size class
// that E1 showed is bit-stable on both backends, so a mismatch versus the reference indicates a state error.
// The peer sequence checks that rollback of sequence 1 leaves sequence 0 untouched.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static const std::vector<int> T = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};
static const std::vector<int> P = {760, 1118, 28279, 9944, 4947, 513, 220, 17, 11, 220, 18};
static const std::vector<int> W = {9927, 279, 1287};
static constexpr int START = 7, TGT = 1, PEER = 0;
using vec = std::vector<float>;

struct item { int seq, pos, id; };
static vec decode(llama_context *c, const std::vector<item> &xs, bool want) {
    auto *b = llama_batch_ext_init(c);
    for (size_t k = 0; k < xs.size(); k++) {
        const bool out = want && k + 1 == xs.size();
        int i = llama_batch_ext_add_token(b, xs[k].seq, xs[k].id);
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
static llama_context *make(llama_model *m, int cap) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 8;
    cp.n_seq_max = 2; cp.n_rs_seq = cap; cp.kv_unified = false;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    return c;
}
static void prefix(llama_context *c) {
    for (int i = 0; i < START; i++) decode(c, {{TGT, i, T[i]}}, false);
    for (int i = 0; i < START + 1; i++) decode(c, {{PEER, i, P[i]}}, false);
}
static vec finish_peer(llama_context *c) {
    vec z;
    for (int i = START + 1; i < (int) P.size(); i++) z = decode(c, {{PEER, i, P[i]}}, i + 1 == (int) P.size());
    return z;
}
static vec finish_target(llama_context *c, int from) {
    vec z;
    for (int i = from; i < (int) T.size(); i++) z = decode(c, {{TGT, i, T[i]}}, i + 1 == (int) T.size());
    return z;
}
static void speculate(llama_context *c, const std::vector<item> &xs, bool batch) {
    if (batch) decode(c, xs, false);
    else for (const auto &x : xs) decode(c, {x}, false);
}
static double maxdiff(const vec &a, const vec &b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) d = std::max<double>(d, std::fabs(a[i] - b[i]));
    return d;
}
static void row(const char *label, const std::string &name, int cycles, bool rm_ok, const vec &t, const vec &p, const vec &rt, const vec &rp) {
    printf("%s,%s,%d,%d,%d,%d,%.8g,%.8g\n", label, name.c_str(), cycles, rm_ok,
           !memcmp(t.data(), rt.data(), t.size() * sizeof(float)), !memcmp(p.data(), rp.data(), p.size() * sizeof(float)),
           maxdiff(t, rt), maxdiff(p, rp));
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const char *label = argv[3];
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;

    vec rt, rp;
    {
        auto *c = make(m, 3);
        prefix(c);
        rt = finish_target(c, START);
        rp = finish_peer(c);
        llama_free(c);
    }
    printf("backend,scenario,rollback_cycles,seq_rm_ok,target_bit_equal,peer_bit_equal,target_max_abs_diff,peer_max_abs_diff\n");

    // single rollback: depth r after a accepted tokens, speculative tokens sent as one batch or one at a time
    for (int shape = 0; shape < 2; shape++) for (int r = 1; r <= 3; r++) for (int a = 0; a + r <= 3; a++) {
        auto *c = make(m, 3);
        prefix(c);
        std::vector<item> xs;
        for (int i = 0; i < a; i++) xs.push_back({TGT, START + i, T[START + i]});
        for (int i = 0; i < r; i++) xs.push_back({TGT, START + a + i, W[i]});
        speculate(c, xs, shape == 0);
        const bool ok = llama_memory_seq_rm(llama_get_memory(c), TGT, START + a, -1);
        vec t = ok ? finish_target(c, START + a) : vec(rt.size(), NAN);
        vec p = finish_peer(c);
        row(label, std::string(shape ? "singles" : "batch") + "-r" + std::to_string(r) + "-a" + std::to_string(a), 1, ok, t, p, rt, rp);
        llama_free(c);
    }

    // repeated rejection: before each true token, speculate 3 wrong tokens and roll all 3 back
    for (int shape = 0; shape < 2; shape++) {
        auto *c = make(m, 3);
        prefix(c);
        bool ok = true;
        vec t;
        for (int q = START; q < (int) T.size(); q++) {
            std::vector<item> xs;
            for (int i = 0; i < 3; i++) xs.push_back({TGT, q + i, W[i]});
            speculate(c, xs, shape == 0);
            ok = llama_memory_seq_rm(llama_get_memory(c), TGT, q, -1) && ok;
            t = decode(c, {{TGT, q, T[q]}}, q + 1 == (int) T.size());
        }
        vec p = finish_peer(c);
        row(label, std::string("repeat-") + (shape ? "singles" : "batch"), (int) T.size() - START, ok, t, p, rt, rp);
        llama_free(c);
    }

    // cost: one-token rollback versus a partial-state checkpoint save and restore
    for (int cap : {0, 3}) {
        auto *c = make(m, cap);
        prefix(c);
        const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        std::vector<double> save_us, load_us, rm_us;
        size_t bytes = 0;
        for (int i = 0; i < 20; i++) {
            auto t0 = clk::now();
            std::vector<uint8_t> s(llama_state_seq_get_size_ext(c, TGT, flags));
            if (llama_state_seq_get_data_ext(c, s.data(), s.size(), TGT, flags) != s.size()) throw std::runtime_error("save");
            auto t1 = clk::now();
            bytes = s.size();
            decode(c, {{TGT, START, W[0]}}, false);
            auto t2 = clk::now();
            if (cap > 0) {
                if (!llama_memory_seq_rm(llama_get_memory(c), TGT, START, -1)) throw std::runtime_error("rm");
                rm_us.push_back(std::chrono::duration<double, std::micro>(clk::now() - t2).count());
                decode(c, {{TGT, START, W[0]}}, false);
                t2 = clk::now();
            }
            if (llama_state_seq_set_data_ext(c, s.data(), s.size(), TGT, flags) != s.size()) throw std::runtime_error("load");
            if (!llama_memory_seq_rm(llama_get_memory(c), TGT, START, -1)) throw std::runtime_error("trim");
            auto t3 = clk::now();
            save_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            load_us.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
        }
        auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        fprintf(stderr, "COST %s cap=%d checkpoint_bytes=%zu save_us=%.0f restore_us=%.0f seq_rm_us=%.1f\n", label, cap, bytes, med(save_us), med(load_us), med(rm_us));
        llama_free(c);
    }
    llama_model_free(m);
    llama_backend_free();
}
