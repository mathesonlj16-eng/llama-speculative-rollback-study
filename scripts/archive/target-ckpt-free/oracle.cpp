// Grouping-aware oracle. Teacher-forces each generated sequence one token at a time on the target model.
// A token is exact if it is the reference argmax, near-tie if the reference prefers another token by at most
// the tolerance, and a hard mismatch otherwise.
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static const float * step(llama_context * c, int id, int pos) {
    auto * b = llama_batch_ext_init(c);
    int i = llama_batch_ext_add_token(b, 0, id);
    if (i < 0 || !llama_batch_ext_set_pos(b, i, &pos) || !llama_batch_ext_set_output_logits(b, i, true)) throw std::runtime_error("batch");
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    auto * z = llama_get_logits_ith(c, 0);
    llama_batch_ext_free(b);
    return z;
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const float tol = std::atof(argv[3]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    std::ifstream in(argv[2]);
    std::string line;
    printf("key,n_tokens,exact,near_tie,hard_mismatch,max_gap,first_hard_index\n");
    while (std::getline(in, line)) {
        auto t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
        const auto key = line.substr(0, t1);
        const auto prompt = ints(line.substr(t1 + 1, t2 - t1 - 1)), gen = ints(line.substr(t2 + 1));
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 8; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        auto * c = llama_init_from_model(m, cp);
        if (!c) return 1;
        const float * z = nullptr;
        int pos = 0;
        for (int id : prompt) z = step(c, id, pos++);
        int exact = 0, near = 0, hard = 0, first = -1;
        double max_gap = 0;
        for (size_t i = 0; i < gen.size(); i++) {
            const int best = std::max_element(z, z + n_vocab) - z;
            const double gap = z[best] - z[gen[i]];
            max_gap = std::max(max_gap, gap);
            if (best == gen[i]) exact++;
            else if (gap <= tol) near++;
            else { hard++; if (first < 0) first = i; }
            if (i + 1 < gen.size()) z = step(c, gen[i], pos++);
        }
        printf("%s,%zu,%d,%d,%d,%.6f,%d\n", key.c_str(), gen.size(), exact, near, hard, max_gap, first);
        fflush(stdout);
        llama_free(c);
    }
    llama_model_free(m);
    llama_backend_free();
}
