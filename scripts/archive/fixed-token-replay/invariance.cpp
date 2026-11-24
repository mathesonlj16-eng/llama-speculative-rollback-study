// Experiment E1: single-sequence chunking invariance.
// Same token IDs, same positions, one sequence, no checkpoint, no peer slot, no draft model.
// Factors varied one at a time: chunking, n_rs_seq (0 or 3), backend (via n_gpu_layers), threads.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static const std::vector<int> TOKENS = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};
static constexpr int SORTING = 27355, ALGORITHM = 11767, SEQ = 1;

struct plan { std::string name; std::vector<int> sizes; };

static std::vector<plan> plans() {
    const int n = TOKENS.size();
    std::vector<plan> out;
    out.push_back({"singles", std::vector<int>(n, 1)});
    out.push_back({"whole", {n}});
    for (int k = 1; k < n; k++) out.push_back({"split" + std::to_string(k), {k, n - k}});
    for (int c : {2, 3, 4}) {
        std::vector<int> s;
        for (int left = n; left > 0; left -= c) s.push_back(std::min(c, left));
        out.push_back({"chunk" + std::to_string(c), s});
    }
    out.push_back({"singles-repeat", std::vector<int>(n, 1)});
    out.push_back({"whole-repeat", {n}});
    return out;
}

static std::vector<float> run(llama_model *m, int cap, int threads, const plan &p) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.n_seq_max = 2; cp.n_rs_seq = cap; cp.kv_unified = false;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
    std::vector<float> result;
    int pos = 0;
    for (int size : p.sizes) {
        auto *b = llama_batch_ext_init(c);
        for (int j = 0; j < size; j++) {
            const int q = pos + j;
            const bool last = q + 1 == (int) TOKENS.size();
            int i = llama_batch_ext_add_token(b, SEQ, TOKENS[q]);
            if (i < 0 || !llama_batch_ext_set_pos(b, i, &q) || !llama_batch_ext_set_output_logits(b, i, last))
                throw std::runtime_error("batch");
        }
        if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
        pos += size;
        if (pos == (int) TOKENS.size()) {
            auto *z = llama_get_logits_ith(c, size - 1);
            if (!z) throw std::runtime_error("logits");
            result.assign(z, z + n_vocab);
        }
        llama_batch_ext_free(b);
    }
    llama_free(c);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: model ngl threads label outdir\n"); return 2; }
    const int ngl = atoi(argv[2]), threads = atoi(argv[3]);
    const std::string label = argv[4], outdir = argv[5];
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    std::vector<float> ref;
    printf("backend,cap,plan,top,margin_sorting_minus_algorithm,max_abs_diff_vs_ref,rms_diff_vs_ref,bit_identical_to_ref\n");
    for (int cap : {0, 3}) for (const auto &p : plans()) {
        auto z = run(m, cap, threads, p);
        if (ref.empty()) ref = z;  // reference: cap 0, one token per decode
        double mx = 0, sq = 0;
        for (size_t i = 0; i < z.size(); i++) { double d = z[i] - ref[i]; mx = std::max(mx, std::fabs(d)); sq += d * d; }
        const bool same = std::equal(z.begin(), z.end(), ref.begin(), [](float a, float b) {
            return std::memcmp(&a, &b, sizeof a) == 0; });
        printf("%s,%d,%s,%td,%.6f,%.6f,%.6f,%d\n", label.c_str(), cap, p.name.c_str(),
               std::max_element(z.begin(), z.end()) - z.begin(), z[SORTING] - z[ALGORITHM], mx,
               std::sqrt(sq / z.size()), same);
        fflush(stdout);
        std::ofstream(outdir + "/" + label + "-cap" + std::to_string(cap) + "-" + p.name + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char *>(z.data()), z.size() * sizeof(float));
    }
    llama_model_free(m);
    llama_backend_free();
}
