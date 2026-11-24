// v1.1.0: teacher-forced CUDA logit dump for cross-build comparison.
// Input TSV: key \t prompt ids \t generated ids. Writes, per sequence in TSV order, the full logits that predict
// each generated token. Shapes: singles (one token per decode); whole (prompt+gen in one decode);
// c4 / c8 (prompt whole, then generated tokens in chunks of 4 / 8); pair4 (TSV rows 0+1, 2+3, ... decoded
// together as two sequences: both prompts in one batch, then 4 tokens of each per batch).
#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct seq_t { std::string key; std::vector<int> prompt, gen; };
static std::vector<int> ints(const std::string & s) { std::istringstream in(s); std::vector<int> v; int x; while (in >> x) v.push_back(x); return v; }
struct item { int seq; int pos; int tok; };
int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: model ngl tsv shape out.bin\n"); return 2; }
    const int ngl = atoi(argv[2]); const std::string shape = argv[4];
    std::vector<seq_t> seqs; std::ifstream in(argv[3]); std::string line;
    while (std::getline(in, line)) { auto a = line.find('\t'), b = line.find('\t', a + 1);
        seqs.push_back({line.substr(0, a), ints(line.substr(a + 1, b - a - 1)), ints(line.substr(b + 1))}); }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    auto * m = llama_model_load_from_file(argv[1], mp); if (!m) return 1;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const bool pair = shape == "pair4";
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_seq_max = pair ? 2 : 1; cp.n_rs_seq = getenv("N_RS") ? (uint32_t) atoi(getenv("N_RS")) : 0;
    cp.n_threads = cp.n_threads_batch = 8;
    FILE * f = fopen(argv[5], "wb"); if (!f) return 1;
    const size_t group = pair ? 2 : 1;
    for (size_t g = 0; g < seqs.size(); g += group) {
        auto * c = llama_init_from_model(m, cp); if (!c) throw std::runtime_error("context");
        const size_t ns = std::min(group, seqs.size() - g);
        std::vector<std::vector<int>> all(ns); std::vector<std::vector<float>> logits(ns);
        for (size_t s = 0; s < ns; s++) { all[s] = seqs[g + s].prompt; all[s].insert(all[s].end(), seqs[g + s].gen.begin(), seqs[g + s].gen.end() - 1); }
        std::vector<std::vector<item>> batches;
        auto span = [&](int s, int b0, int e0, std::vector<item> & v) { for (int q = b0; q < e0; q++) v.push_back({s, q, all[s][q]}); };
        if (shape == "singles") for (int q = 0; q < (int) all[0].size(); q++) { batches.emplace_back(); span(0, q, q + 1, batches.back()); }
        else if (shape == "whole") { batches.emplace_back(); span(0, 0, all[0].size(), batches.back()); }
        else if (shape == "c4" || shape == "c8" || pair) {
            const int k = shape == "c8" ? 8 : 4; batches.emplace_back();
            for (size_t s = 0; s < ns; s++) span(s, 0, seqs[g + s].prompt.size(), batches.back());
            for (int off = 0;; off += k) { std::vector<item> v;
                for (size_t s = 0; s < ns; s++) { const int n = seqs[g + s].prompt.size(), t = all[s].size(); span(s, std::min(t, n + off), std::min(t, n + off + k), v); }
                if (v.empty()) break; batches.push_back(v); }
        } else throw std::runtime_error("shape");
        for (auto & v : batches) {
            auto * b = llama_batch_ext_init(c); std::vector<int> want;
            for (auto & it : v) { const bool out = it.pos >= (int) seqs[g + it.seq].prompt.size() - 1;
                int i = llama_batch_ext_add_token(b, it.seq, it.tok);
                if (i < 0 || !llama_batch_ext_set_pos(b, i, &it.pos) || !llama_batch_ext_set_output_logits(b, i, out)) throw std::runtime_error("batch");
                if (out) want.push_back(i); }
            if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
            size_t w = 0;
            for (auto & it : v) if (it.pos >= (int) seqs[g + it.seq].prompt.size() - 1) {
                const float * z = llama_get_logits_ith(c, want[w++]); logits[it.seq].insert(logits[it.seq].end(), z, z + nv); }
            llama_batch_ext_free(b);
        }
        for (size_t s = 0; s < ns; s++) { int32_t hdr[2] = {(int32_t) seqs[g + s].gen.size(), nv}; fwrite(hdr, sizeof hdr, 1, f);
            if (logits[s].size() != (size_t) hdr[0] * nv) throw std::runtime_error("count"); fwrite(logits[s].data(), 4, logits[s].size(), f); }
        llama_free(c);
    }
    fclose(f); llama_model_free(m); llama_backend_free();
}
