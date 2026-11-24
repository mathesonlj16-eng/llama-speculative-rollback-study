// G5: locates the operation where a perturbation stops scaling with its size. Restores the state before one decode,
// perturbs l_out-18 by a relative size, and compares every later F32 tensor with an unperturbed capture.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vec = std::vector<float>;
struct rec { std::string key, op, shape; vec v; };
struct cap_t {
    bool active = false, after = false;
    double eps = 0;
    unsigned seed = 1;
    std::map<std::string, int> seen;
    std::vector<rec> rows;
};

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static bool on_eval(ggml_tensor * t, bool ask, void * u) {
    auto * c = static_cast<cap_t *>(u);
    if (ask) return c->active;
    if (!c->active || t->type != GGML_TYPE_F32 || ggml_nelements(t) > 1000000 || ggml_nelements(t) == 0) return true;
    const std::string name = ggml_get_name(t);
    const bool inject = name == "l_out-18";
    if (inject) {
        c->after = true;
        if (c->eps > 0) {
            vec v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
            std::mt19937 rng(c->seed);
            for (auto & x : v) x = (float) (x * (1.0 + ((rng() & 1) ? c->eps : -c->eps)));
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
    if (!c->after) return true;
    rec r;
    r.key = name + "#" + std::to_string(c->seen[name]++);
    r.op = ggml_op_name(t->op);
    r.shape = std::to_string(t->ne[0]) + "x" + std::to_string(t->ne[1]) + "x" + std::to_string(t->ne[2]) + "x" + std::to_string(t->ne[3]);
    r.v.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, r.v.data(), 0, r.v.size() * sizeof(float));
    c->rows.push_back(std::move(r));
    return true;
}
static void step(llama_context * c, int id, int pos) {
    auto * b = llama_batch_ext_init(c);
    int i = llama_batch_ext_add_token(b, 0, id);
    if (i < 0 || !llama_batch_ext_set_pos(b, i, &pos) || !llama_batch_ext_set_output_logits(b, i, true)) throw std::runtime_error("batch");
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    llama_get_logits_ith(c, 0);
    llama_batch_ext_free(b);
}
static double rms_diff(const vec & a, const vec & b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) { const double d = (double) a[i] - b[i]; s += d * d; }
    return std::sqrt(s / a.size());
}

int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: model tsv kv flash label\n"); return 2; }
    const std::string kv = argv[3], flash = argv[4], label = argv[5];
    std::vector<int> prompt, gen;
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("seq15\t", 0) != 0) continue;
        auto a = line.find('\t'), b = line.find('\t', a + 1);
        prompt = ints(line.substr(a + 1, b - a - 1));
        gen = ints(line.substr(b + 1));
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    cap_t cap;
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.n_threads = cp.n_threads_batch = 8; cp.cb_eval = on_eval; cp.cb_eval_user_data = &cap;
    cp.type_k = cp.type_v = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.flash_attn_type = flash == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    auto * c = llama_init_from_model(m, cp);
    if (!c) return 1;
    std::vector<int> all = prompt;
    all.insert(all.end(), gen.begin(), gen.end());
    const int input_pos = (int) prompt.size() + 30 - 1;
    for (int p = 0; p < input_pos; p++) step(c, all[p], p);
    std::vector<uint8_t> st(llama_state_seq_get_size_ext(c, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    if (llama_state_seq_get_data_ext(c, st.data(), st.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st.size()) throw std::runtime_error("save");
    auto capture = [&](double eps) {
        llama_memory_seq_rm(llama_get_memory(c), 0, -1, -1);
        if (llama_state_seq_set_data_ext(c, st.data(), st.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != st.size()) throw std::runtime_error("load");
        cap = {};
        cap.active = true;
        cap.eps = eps;
        step(c, all[input_pos], input_pos);
        cap.active = false;
        return std::move(cap.rows);
    };
    const auto base = capture(0), control = capture(0);
    const std::vector<double> sizes = {1e-6, 1e-5, 1e-4, 1e-3};
    std::vector<std::vector<rec>> runs;
    for (double e : sizes) runs.push_back(capture(e));
    printf("config,order,tensor,op,shape,control_diff,d_1e-6,d_1e-5,d_1e-4,d_1e-3,scale_1e-3_over_1e-6\n");
    for (size_t i = 0; i < base.size(); i++) {
        std::vector<double> d;
        bool ok = control.size() == base.size() && control[i].key == base[i].key;
        for (const auto & r : runs) {
            ok = ok && i < r.size() && r[i].key == base[i].key && r[i].v.size() == base[i].v.size();
            d.push_back(ok ? rms_diff(r[i].v, base[i].v) : NAN);
        }
        if (!ok) continue;
        printf("%s,%zu,%s,%s,%s,%.4g,%.4g,%.4g,%.4g,%.4g,%.4g\n", label.c_str(), i, base[i].key.c_str(), base[i].op.c_str(), base[i].shape.c_str(),
               rms_diff(control[i].v, base[i].v), d[0], d[1], d[2], d[3], d[0] > 0 ? d[3] / d[0] : 0.0);
    }
    llama_free(c);
    llama_model_free(m);
    llama_backend_free();
}
