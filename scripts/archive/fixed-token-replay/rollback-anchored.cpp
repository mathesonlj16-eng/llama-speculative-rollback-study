// Experiment E8: checkpoint-free rollback using the server's verification batch shape.
// Each speculative batch is [accepted anchor token, d draft tokens]. Rollback keeps the anchor plus a accepted
// draft tokens. Results are compared bit-for-bit with a never-speculated run, for the target and a peer sequence.
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
using vec = std::vector<float>;
static const std::vector<int> T = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};
static const std::vector<int> P = {760, 1118, 28279, 9944, 4947, 513, 220, 17, 11, 220, 18};
static const std::vector<int> W = {9927, 279, 1287, 314};
static constexpr int START = 7, TGT = 1, PEER = 0;
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
    for (int i = 0; i <= START; i++) decode(c, {{PEER, i, P[i]}}, false);
}
static vec finish(llama_context *c, const std::vector<int> &s, int seq, int from) {
    vec z;
    for (int i = from; i < (int) s.size(); i++) z = decode(c, {{seq, i, s[i]}}, i + 1 == (int) s.size());
    return z;
}
static double maxdiff(const vec &a, const vec &b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) d = std::max<double>(d, std::fabs(a[i] - b[i]));
    return d;
}
static void row(const char *label, const std::string &name, int cycles, bool ok, const vec &t, const vec &p, const vec &rt, const vec &rp) {
    if (!ok) { printf("%s,%s,%d,0,NA,NA,NA,NA\n", label, name.c_str(), cycles); fflush(stdout); return; }
    printf("%s,%s,%d,1,%d,%d,%.8g,%.8g\n", label, name.c_str(), cycles,
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
        rt = finish(c, T, TGT, START);
        rp = finish(c, P, PEER, START + 1);
        llama_free(c);
    }
    printf("backend,scenario,rollback_cycles,seq_rm_ok,target_bit_equal,peer_bit_equal,target_max_abs_diff,peer_max_abs_diff\n");

    for (int d = 1; d <= 3; d++) for (int a = 0; a < d; a++) {
        const int r = d - a;
        auto *c = make(m, 3);
        prefix(c);
        std::vector<item> xs{{TGT, START, T[START]}};
        for (int i = 0; i < a; i++) xs.push_back({TGT, START + 1 + i, T[START + 1 + i]});
        for (int i = 0; i < r; i++) xs.push_back({TGT, START + 1 + a + i, W[i]});
        decode(c, xs, false);
        const bool ok = llama_memory_seq_rm(llama_get_memory(c), TGT, START + 1 + a, -1);
        vec t, p;
        if (ok) { t = finish(c, T, TGT, START + 1 + a); p = finish(c, P, PEER, START + 1); }
        row(label, "anchored-d" + std::to_string(d) + "-a" + std::to_string(a) + "-r" + std::to_string(r), 1, ok, t, p, rt, rp);
        llama_free(c);
    }
    {   // repeated full rejection of a 3-token draft before each accepted token
        auto *c = make(m, 3);
        prefix(c);
        bool ok = true;
        int cycles = 0;
        for (int q = START; q < (int) T.size() - 1 && ok; q++, cycles++) {
            std::vector<item> xs{{TGT, q, T[q]}};
            for (int i = 0; i < 3; i++) xs.push_back({TGT, q + 1 + i, W[i]});
            decode(c, xs, false);
            ok = llama_memory_seq_rm(llama_get_memory(c), TGT, q + 1, -1);
        }
        vec t, p;
        if (ok) { t = finish(c, T, TGT, (int) T.size() - 1); p = finish(c, P, PEER, START + 1); }
        row(label, "repeat-full-reject-d3", cycles, ok, t, p, rt, rp);
        llama_free(c);
    }
    {   // negative control: rollback deeper than n_rs_seq must be refused
        auto *c = make(m, 3);
        prefix(c);
        std::vector<item> xs{{TGT, START, T[START]}};
        for (int i = 0; i < 4; i++) xs.push_back({TGT, START + 1 + i, W[i]});
        decode(c, xs, false);
        const bool ok = llama_memory_seq_rm(llama_get_memory(c), TGT, START + 1, -1);
        printf("%s,over-limit-r4-expected-refusal,1,%d,NA,NA,NA,NA\n", label, ok);
        fflush(stdout);
        llama_free(c);
    }
    for (int cap : {0, 3}) {
        auto *c = make(m, cap);
        prefix(c);
        const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        std::vector<double> save, load, rm;
        size_t bytes = 0;
        int q = START;
        for (int i = 0; i < 20; i++) {
            auto t0 = clk::now();
            std::vector<uint8_t> s(llama_state_seq_get_size_ext(c, TGT, flags));
            if (llama_state_seq_get_data_ext(c, s.data(), s.size(), TGT, flags) != s.size()) throw std::runtime_error("save");
            auto t1 = clk::now();
            if (llama_state_seq_set_data_ext(c, s.data(), s.size(), TGT, flags) != s.size()) throw std::runtime_error("load");
            auto t2 = clk::now();
            bytes = s.size();
            save.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            load.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
            if (cap > 0) {
                decode(c, {{TGT, q, 11}, {TGT, q + 1, 13}}, false);
                auto t3 = clk::now();
                if (!llama_memory_seq_rm(llama_get_memory(c), TGT, q + 1, -1)) throw std::runtime_error("rm");
                rm.push_back(std::chrono::duration<double, std::micro>(clk::now() - t3).count());
                q++;
            }
        }
        auto med = [](std::vector<double> v) { if (v.empty()) return -1.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        fprintf(stderr, "COST %s cap=%d checkpoint_bytes=%zu save_us=%.0f restore_us=%.0f seq_rm_us=%.2f\n", label, cap, bytes, med(save), med(load), med(rm));
        llama_free(c);
    }
    llama_model_free(m);
    llama_backend_free();
}
