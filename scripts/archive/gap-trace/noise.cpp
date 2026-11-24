// G3: noise floor. Decodes one token at a time on CPU and moves every element of each layer output by one ULP in a
// random direction. Batch shape never changes, so any logit shift here comes from rounding-scale perturbation.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vec = std::vector<float>;
struct seq_t { std::string key; std::vector<int> prompt, gen; };
struct noise_t { bool on = false; std::mt19937 rng; };

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static bool on_eval(ggml_tensor * t, bool ask, void * u) {
    auto * n = static_cast<noise_t *>(u);
    const bool hit = n->on && t->type == GGML_TYPE_F32 && !strncmp(ggml_get_name(t), "l_out-", 6);
    if (ask) return hit;
    if (!hit) return true;
    vec v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    for (auto & x : v) x = std::nextafter(x, (n->rng() & 1) ? INFINITY : -INFINITY);
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    return true;
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
static std::vector<vec> run(llama_model * m, const seq_t & s, noise_t & noise) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = on_eval; cp.cb_eval_user_data = &noise;
    auto * c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    const float * z = nullptr;
    int pos = 0;
    for (int id : s.prompt) z = step(c, id, pos++);
    std::vector<vec> out;
    out.emplace_back(z, z + n_vocab);
    for (size_t i = 0; i + 1 < s.gen.size(); i++) {
        z = step(c, s.gen[i], pos++);
        out.emplace_back(z, z + n_vocab);
    }
    llama_free(c);
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const std::set<std::string> keep = {"seq8", "seq15"};
    std::vector<seq_t> seqs;
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        auto a = line.find('\t'), b = line.find('\t', a + 1);
        seq_t s{line.substr(0, a), ints(line.substr(a + 1, b - a - 1)), ints(line.substr(b + 1))};
        if (keep.count(s.key)) seqs.push_back(std::move(s));
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    printf("key,seed,index,gen_token,ref_top,top,gen_gap,max_abs_diff,rms_diff\n");
    for (const auto & s : seqs) {
        noise_t off;
        const auto ref = run(m, s, off);
        for (unsigned seed : {0u, 1u, 2u, 3u, 4u}) {
            noise_t n;
            n.on = seed != 0;   // seed 0 is the no-noise control and must be bit-identical
            n.rng.seed(seed);
            const auto r = run(m, s, n);
            for (size_t i = 0; i < r.size(); i++) {
                double mx = 0, sq = 0;
                for (size_t k = 0; k < r[i].size(); k++) { const double d = r[i][k] - ref[i][k]; mx = std::max(mx, std::fabs(d)); sq += d * d; }
                const int top = std::max_element(r[i].begin(), r[i].end()) - r[i].begin();
                const int rtop = std::max_element(ref[i].begin(), ref[i].end()) - ref[i].begin();
                printf("%s,%u,%zu,%d,%d,%d,%.6f,%.6f,%.6f\n", s.key.c_str(), seed, i, s.gen[i], rtop, top, r[i][top] - r[i][s.gen[i]], mx, std::sqrt(sq / r[i].size()));
            }
            fflush(stdout);
        }
    }
    llama_model_free(m);
    llama_backend_free();
}
