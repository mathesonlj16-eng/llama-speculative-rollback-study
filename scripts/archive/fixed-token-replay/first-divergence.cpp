// Experiment E2: locate the first graph tensor whose value for the final token differs
// between two chunkings of identical tokens. Single sequence, n_rs_seq=0, no checkpoint.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const std::vector<int> TOKENS = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};

struct rec {
    std::string key, op;
    std::array<int64_t, 4> ne;
    int axis;
    std::vector<float> v;
};
struct capture { bool active = false; int n_tokens = 0; std::vector<rec> rows; std::map<std::string, int> seen; };

static bool on_eval(ggml_tensor *t, bool ask, void *u) {
    if (ask) return true;
    auto *c = static_cast<capture *>(u);
    if (!c->active || t->type != GGML_TYPE_F32 || ggml_nelements(t) > 8000000) return true;
    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, all.size() * sizeof(float));
    std::array<int64_t, 4> ne{t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    int axis = -1;
    for (int a : {2, 1, 3, 0}) if (ne[a] == c->n_tokens) { axis = a; break; }
    rec r;
    const std::string name = ggml_get_name(t);
    r.key = name + "#" + std::to_string(c->seen[name]++);
    r.op = ggml_op_name(t->op); r.ne = ne; r.axis = axis;
    if (axis < 0) r.v = std::move(all);
    else {
        int64_t stride = 1;
        for (int a = 0; a < axis; a++) stride *= ne[a];
        for (int64_t i = 0; i < (int64_t) all.size(); i++)
            if ((i / stride) % ne[axis] == c->n_tokens - 1) r.v.push_back(all[i]);
    }
    c->rows.push_back(std::move(r));
    return true;
}

static std::vector<float> run(llama_model *m, const std::vector<int> &sizes, capture &cap) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 1;
    cp.n_seq_max = 2; cp.n_rs_seq = 0; cp.kv_unified = false;
    cp.cb_eval = on_eval; cp.cb_eval_user_data = &cap;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    std::vector<float> logits;
    int pos = 0;
    for (size_t s = 0; s < sizes.size(); s++) {
        cap.active = s + 1 == sizes.size();
        cap.n_tokens = sizes[s];
        auto *b = llama_batch_ext_init(c);
        for (int j = 0; j < sizes[s]; j++) {
            const int q = pos + j;
            const bool last = q + 1 == (int) TOKENS.size();
            int i = llama_batch_ext_add_token(b, 1, TOKENS[q]);
            if (i < 0 || !llama_batch_ext_set_pos(b, i, &q) || !llama_batch_ext_set_output_logits(b, i, last))
                throw std::runtime_error("batch");
        }
        if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
        pos += sizes[s];
        if (cap.active) {
            auto *z = llama_get_logits_ith(c, sizes[s] - 1);
            logits.assign(z, z + llama_vocab_n_tokens(llama_model_get_vocab(m)));
        }
        llama_batch_ext_free(b);
    }
    llama_free(c);
    return logits;
}

static std::string shape(const rec &r) {
    std::ostringstream o;
    o << r.ne[0] << 'x' << r.ne[1] << 'x' << r.ne[2] << 'x' << r.ne[3] << "/axis" << r.axis;
    return o.str();
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    capture a, b;
    auto la = run(m, {8, 3}, a);
    auto lb = run(m, {7, 4}, b);
    double logit_max = 0;
    for (size_t i = 0; i < la.size(); i++) logit_max = std::max<double>(logit_max, std::fabs(la[i] - lb[i]));
    fprintf(stderr, "SUMMARY final_logit_max_abs_diff=%.6f margin_a=%.6f margin_b=%.6f nodes_a=%zu nodes_b=%zu\n",
            logit_max, la[27355] - la[11767], lb[27355] - lb[11767], a.rows.size(), b.rows.size());
    std::map<std::string, const rec *> by_key;
    for (const auto &r : a.rows) by_key[r.key] = &r;
    printf("order,tensor,op,shape_8+3,shape_7+4,comparable,max_abs_diff,max_abs_value\n");
    for (size_t i = 0; i < b.rows.size(); i++) {
        const auto &y = b.rows[i];
        auto it = by_key.find(y.key);
        if (it == by_key.end()) { printf("%zu,%s,%s,missing,%s,0,,\n", i, y.key.c_str(), y.op.c_str(), shape(y).c_str()); continue; }
        const auto &x = *it->second;
        if (x.v.size() != y.v.size()) { printf("%zu,%s,%s,%s,%s,0,,\n", i, y.key.c_str(), y.op.c_str(), shape(x).c_str(), shape(y).c_str()); continue; }
        double mx = 0, mag = 0;
        for (size_t k = 0; k < x.v.size(); k++) { mx = std::max<double>(mx, std::fabs(x.v[k] - y.v[k])); mag = std::max<double>(mag, std::fabs(y.v[k])); }
        printf("%zu,%s,%s,%s,%s,1,%.8g,%.8g\n", i, y.key.c_str(), y.op.c_str(), shape(x).c_str(), shape(y).c_str(), mx, mag);
    }
    llama_model_free(m);
    llama_backend_free();
}
