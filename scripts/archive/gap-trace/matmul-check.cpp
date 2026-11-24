// G10: checks the first layer-0 matrix multiply (node_15) against an independent double-precision reference.
// For one-token and whole-prompt decoding of the same prompt, captures the final token's input row and output row,
// dequantizes every weight row with ggml's own type traits, and recomputes the product in double precision.
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct cap_t { bool active = false; int n_tokens = 0; std::vector<float> in, out; double max_rel = 0, max_abs = 0; std::string info; };

static std::vector<int> ints(const std::string & s) {
    std::istringstream in(s);
    std::vector<int> v;
    int x;
    while (in >> x) v.push_back(x);
    return v;
}
static std::vector<float> row(const ggml_tensor * t, int64_t r) {
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("expected F32");
    std::vector<float> v(t->ne[0]);
    ggml_backend_tensor_get(t, v.data(), r * t->nb[1], t->ne[0] * sizeof(float));
    return v;
}
static bool on_eval(ggml_tensor * t, bool ask, void * u) {
    auto * c = static_cast<cap_t *>(u);
    const bool hit = c->active && t->op == GGML_OP_MUL_MAT && !strcmp(ggml_get_name(t), "node_15");
    if (ask) return hit;
    if (!hit) return true;
    const ggml_tensor * w = t->src[0], * x = t->src[1];
    const int64_t last = c->n_tokens - 1;
    c->in = row(x, last);
    c->out = row(t, last);
    char buf[512];
    snprintf(buf, sizeof buf, "weight=%s type=%s ne=%ldx%ld extra=%d | input=%s type=%s ne=%ldx%ld contiguous=%d | hint=%d",
             ggml_get_name(w), ggml_type_name(w->type), (long) w->ne[0], (long) w->ne[1], w->extra != nullptr,
             ggml_get_name(x), ggml_type_name(x->type), (long) x->ne[0], (long) x->ne[1], ggml_is_contiguous(x), t->op_params[1]);
    c->info = buf;
    std::vector<uint8_t> raw(ggml_nbytes(w));
    ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
    std::vector<float> wr(w->ne[0]);
    const auto * tr = ggml_get_type_traits(w->type);
    for (int64_t r = 0; r < w->ne[1]; r++) {
        tr->to_float(raw.data() + r * w->nb[1], wr.data(), w->ne[0]);
        double ref = 0, mag = 0;
        for (int64_t k = 0; k < w->ne[0]; k++) { ref += (double) wr[k] * c->in[k]; mag += std::fabs((double) wr[k] * c->in[k]); }
        const double err = std::fabs(c->out[r] - ref);
        c->max_abs = std::max(c->max_abs, err);
        c->max_rel = std::max(c->max_rel, mag > 0 ? err / mag : 0);
    }
    return true;
}
static void decode(llama_context * ctx, const std::vector<int> & ids, int pos0) {
    auto * b = llama_batch_ext_init(ctx);
    for (size_t k = 0; k < ids.size(); k++) {
        const int q = pos0 + (int) k;
        int i = llama_batch_ext_add_token(b, 0, ids[k]);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &q) || !llama_batch_ext_set_output_logits(b, i, k + 1 == ids.size())) throw std::runtime_error("batch");
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    llama_batch_ext_free(b);
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const std::string label = argv[3];
    std::vector<int> prompt;
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("seq15\t", 0) != 0) continue;
        auto a = line.find('\t'), b = line.find('\t', a + 1);
        prompt = ints(line.substr(a + 1, b - a - 1));
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    std::vector<float> in_a, out_a;
    for (const char * shape : {"singles", "whole"}) {
        cap_t cap;
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_rs_seq = 0; cp.n_threads = cp.n_threads_batch = 8;
        cp.type_k = cp.type_v = GGML_TYPE_F32; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = on_eval; cp.cb_eval_user_data = &cap;
        auto * ctx = llama_init_from_model(m, cp);
        if (!ctx) return 1;
        const int n = (int) prompt.size();
        if (!strcmp(shape, "singles")) {
            for (int p = 0; p < n; p++) {
                cap.active = p == n - 1; cap.n_tokens = 1;
                decode(ctx, {prompt[p]}, p);
            }
        } else {
            cap.active = true; cap.n_tokens = n;
            decode(ctx, prompt, 0);
        }
        llama_free(ctx);
        printf("%s %s %s\n", label.c_str(), shape, cap.info.c_str());
        printf("%s %s versus double-precision reference: max_abs_err=%.6g max_rel_err=%.6g\n", label.c_str(), shape, cap.max_abs, cap.max_rel);
        if (in_a.empty()) { in_a = cap.in; out_a = cap.out; continue; }
        double din = 0, dout = 0;
        for (size_t k = 0; k < in_a.size(); k++) din = std::max(din, (double) std::fabs(in_a[k] - cap.in[k]));
        for (size_t k = 0; k < out_a.size(); k++) dout = std::max(dout, (double) std::fabs(out_a[k] - cap.out[k]));
        printf("%s singles-vs-whole: input max diff=%.6g output max diff=%.6g\n", label.c_str(), din, dout);
    }
    llama_model_free(m);
    llama_backend_free();
}
