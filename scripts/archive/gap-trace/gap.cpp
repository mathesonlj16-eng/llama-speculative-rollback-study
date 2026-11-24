// G1: batch-shape sensitivity on long prompts. Teacher-forces fixed token sequences and compares every
// generated position against the one-token-at-a-time reference for the same backend and cache settings.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vec = std::vector<float>;
struct seq_t { std::string key; std::vector<int> prompt, gen; };

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static void decode(llama_context * c, const std::vector<int> & ids, int pos0, bool want_last) {
    auto * b = llama_batch_ext_init(c);
    for (size_t k = 0; k < ids.size(); k++) {
        const int q = pos0 + (int) k;
        int i = llama_batch_ext_add_token(b, 0, ids[k]);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &q) || !llama_batch_ext_set_output_logits(b, i, want_last && k + 1 == ids.size()))
            throw std::runtime_error("batch");
    }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    llama_batch_ext_free(b);
}
static std::vector<vec> run(llama_model * m, llama_context_params cp, const seq_t & s, int chunk) {
    auto * c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const int n = (int) s.prompt.size();
    const int step = chunk <= 0 ? n : chunk;
    int last = 1;
    for (int p = 0; p < n; p += step) {
        const int e = std::min(n, p + step);
        decode(c, std::vector<int>(s.prompt.begin() + p, s.prompt.begin() + e), p, e == n);
        last = e - p;
    }
    std::vector<vec> out;
    const float * z = llama_get_logits_ith(c, last - 1);
    out.emplace_back(z, z + n_vocab);
    for (size_t i = 0; i + 1 < s.gen.size(); i++) {
        decode(c, {s.gen[i]}, n + (int) i, true);
        z = llama_get_logits_ith(c, 0);
        out.emplace_back(z, z + n_vocab);
    }
    llama_free(c);
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 8) { fprintf(stderr, "usage: model ngl threads kv flash label tsv\n"); return 2; }
    const int ngl = atoi(argv[2]), threads = atoi(argv[3]);
    const std::string kv = argv[4], flash = argv[5], label = argv[6];
    const std::set<std::string> keep = {"seq8", "seq15"};
    std::vector<seq_t> seqs;
    std::ifstream in(argv[7]);
    std::string line;
    while (std::getline(in, line)) {
        auto a = line.find('\t'), b = line.find('\t', a + 1);
        seq_t s{line.substr(0, a), ints(line.substr(a + 1, b - a - 1)), ints(line.substr(b + 1))};
        if (keep.count(s.key)) seqs.push_back(std::move(s));
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.type_k = cp.type_v = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.flash_attn_type = flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    printf("config,key,shape,index,gen_token,ref_top,top,gen_gap,max_abs_diff_vs_singles,rms_diff_vs_singles\n");
    std::vector<std::pair<std::string, int>> shapes = {{"singles", 1}, {"whole", 0}};
    if (ngl > 0) { shapes.push_back({"chunk64", 64}); shapes.push_back({"singles-repeat", 1}); }
    for (const auto & s : seqs) {
        std::vector<vec> ref;
        for (const auto & [name, chunk] : shapes) {
            auto r = run(m, cp, s, chunk);
            if (ref.empty()) ref = r;
            for (size_t i = 0; i < r.size(); i++) {
                double mx = 0, sq = 0;
                for (size_t k = 0; k < r[i].size(); k++) { const double d = r[i][k] - ref[i][k]; mx = std::max(mx, std::fabs(d)); sq += d * d; }
                const int top = std::max_element(r[i].begin(), r[i].end()) - r[i].begin();
                const int rtop = std::max_element(ref[i].begin(), ref[i].end()) - ref[i].begin();
                printf("%s,%s,%s,%zu,%d,%d,%d,%.6f,%.6f,%.6f\n", label.c_str(), s.key.c_str(), name.c_str(), i, s.gen[i], rtop, top,
                       r[i][top] - r[i][s.gen[i]], mx, std::sqrt(sq / r[i].size()));
            }
            fflush(stdout);
        }
    }
    llama_model_free(m);
    llama_backend_free();
}
