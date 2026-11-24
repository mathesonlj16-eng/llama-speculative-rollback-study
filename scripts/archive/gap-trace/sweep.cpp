// G4: perturbation-size sweep on CPU. Restores the exact state before one decode, perturbs one layer output for that
// token, and measures the logit change. Smooth amplification scales with size. A plateau or a few distinct outcomes
// indicates a discontinuous step.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vec = std::vector<float>;
struct seq_t { std::vector<int> prompt, gen; };
struct arm_t { std::string name; double eps = 0; bool ulp = false; unsigned seed = 0; double in_rms = 0; int hits = 0; };

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static bool on_eval(ggml_tensor * t, bool ask, void * u) {
    auto * a = static_cast<arm_t *>(u);
    const bool hit = !a->name.empty() && t->type == GGML_TYPE_F32 && a->name == ggml_get_name(t);
    if (ask) return hit;
    if (!hit) return true;
    vec v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    std::mt19937 rng(a->seed);
    double sq = 0;
    for (auto & x : v) {
        const float old = x;
        const bool up = rng() & 1;
        x = a->ulp ? std::nextafter(x, up ? INFINITY : -INFINITY) : (float) (x * (1.0 + (up ? a->eps : -a->eps)));
        sq += (double) (x - old) * (x - old);
    }
    a->in_rms = std::sqrt(sq / v.size());
    a->hits++;
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    return true;
}
static vec step(llama_context * c, int id, int pos) {
    auto * b = llama_batch_ext_init(c);
    int i = llama_batch_ext_add_token(b, 0, id);
    if (i < 0 || !llama_batch_ext_set_pos(b, i, &pos) || !llama_batch_ext_set_output_logits(b, i, true)) throw std::runtime_error("batch");
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    auto * z = llama_get_logits_ith(c, 0);
    vec out(z, z + llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c))));
    llama_batch_ext_free(b);
    return out;
}
static uint64_t fnv(const vec & v) {
    uint64_t h = 1469598103934665603ULL;
    const auto * p = reinterpret_cast<const uint8_t *>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(float); i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    std::map<std::string, seq_t> seqs;
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        auto a = line.find('\t'), b = line.find('\t', a + 1);
        seqs[line.substr(0, a)] = {ints(line.substr(a + 1, b - a - 1)), ints(line.substr(b + 1))};
    }
    const std::vector<std::pair<std::string, int>> sites = {{"seq15", 11}, {"seq15", 30}, {"seq8", 56}};
    const std::vector<int> layers = {0, 6, 12, 18, 23};
    const std::vector<double> sizes = {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3};
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    printf("key,index,layer,size,seed,hits,input_rms,bit_equal,max_abs_diff,rms_diff,top,base_top,hash\n");
    for (const auto & key : {std::string("seq15"), std::string("seq8")}) {
        const auto & s = seqs.at(key);
        arm_t arm;
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 8; cp.cb_eval = on_eval; cp.cb_eval_user_data = &arm;
        auto * c = llama_init_from_model(m, cp);
        if (!c) return 1;
        std::vector<int> all = s.prompt;
        all.insert(all.end(), s.gen.begin(), s.gen.end());
        int done = 0;
        for (const auto & site : sites) {
            if (site.first != key) continue;
            const int input_pos = (int) s.prompt.size() + site.second - 1;
            while (done < input_pos) { step(c, all[done], done); done++; }
            std::vector<uint8_t> st(llama_state_seq_get_size_ext(c, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
            if (llama_state_seq_get_data_ext(c, st.data(), st.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st.size()) throw std::runtime_error("save");
            auto restore = [&] {
                llama_memory_seq_rm(llama_get_memory(c), 0, -1, -1);
                if (llama_state_seq_set_data_ext(c, st.data(), st.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st.size()) throw std::runtime_error("load");
            };
            const vec base = step(c, all[input_pos], input_pos);
            const int base_top = std::max_element(base.begin(), base.end()) - base.begin();
            for (int layer : layers) for (double size : sizes) for (unsigned seed = 1; seed <= (size == 0 ? 1u : 8u); seed++) {
                restore();
                arm = {};
                if (size > 0) { arm.name = "l_out-" + std::to_string(layer); arm.eps = size; arm.seed = seed; }
                const vec z = step(c, all[input_pos], input_pos);
                const int hits = arm.hits;
                const double in_rms = arm.in_rms;
                arm = {};
                double mx = 0, sq = 0;
                for (size_t k = 0; k < z.size(); k++) { const double d = z[k] - base[k]; mx = std::max(mx, std::fabs(d)); sq += d * d; }
                printf("%s,%d,%d,%g,%u,%d,%.6g,%d,%.6f,%.6f,%td,%d,%016llx\n", key.c_str(), site.second, layer, size, seed, hits, in_rms,
                       !memcmp(z.data(), base.data(), z.size() * sizeof(float)), mx, std::sqrt(sq / z.size()),
                       std::max_element(z.begin(), z.end()) - z.begin(), base_top, (unsigned long long) fnv(z));
            }
            fflush(stdout);
            restore();
        }
        llama_free(c);
    }
    llama_model_free(m);
    llama_backend_free();
}
