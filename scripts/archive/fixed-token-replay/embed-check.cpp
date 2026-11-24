// Experiment E4: validate the E2/E3 capture. For every GET_ROWS node in each decode, print the
// source tensor, the row indices actually used, and a hash of each output row.
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static const std::vector<int> TOKENS = {32, 6165, 7785, 53235, 14330, 27355, 430, 10695, 25, 328, 760};
static std::string g_run;
static int g_decode = 0;

static uint64_t fnv(const void *p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= static_cast<const uint8_t *>(p)[i]; h *= 1099511628211ULL; }
    return h;
}

static bool on_eval(ggml_tensor *t, bool ask, void *) {
    if (t->op != GGML_OP_GET_ROWS) return ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || !t->src[0] || !t->src[1]) return true;
    const std::string src = ggml_get_name(t->src[0]);
    if (src.find("token_embd") == std::string::npos && t->ne[0] != 2048) return true;
    std::vector<int32_t> idx(ggml_nelements(t->src[1]));
    if (t->src[1]->type == GGML_TYPE_I32) ggml_backend_tensor_get(t->src[1], idx.data(), 0, idx.size() * sizeof(int32_t));
    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, all.size() * sizeof(float));
    printf("%s decode=%d node=%s src=%s src_type=%s idx_type=%s shape=%ldx%ld\n", g_run.c_str(), g_decode, ggml_get_name(t), src.c_str(),
           ggml_type_name(t->src[0]->type), ggml_type_name(t->src[1]->type), (long) t->ne[0], (long) t->ne[1]);
    for (int64_t r = 0; r < t->ne[1]; r++)
        printf("  row=%ld index=%d hash=%016llx first=%.6f\n", (long) r, r < (int64_t) idx.size() ? idx[r] : -1,
               (unsigned long long) fnv(all.data() + r * t->ne[0], t->ne[0] * sizeof(float)), all[r * t->ne[0]]);
    return true;
}

static void run(llama_model *m, const std::string &name, const std::vector<int> &sizes) {
    g_run = name;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 1;
    cp.n_seq_max = 2; cp.n_rs_seq = 0; cp.kv_unified = false; cp.cb_eval = on_eval;
    auto *c = llama_init_from_model(m, cp);
    if (!c) throw std::runtime_error("context");
    int pos = 0;
    for (size_t s = 0; s < sizes.size(); s++) {
        g_decode = s;
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
        llama_batch_ext_free(b);
    }
    llama_free(c);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    run(m, "A:7+1+3", {7, 1, 3});
    run(m, "B:7+4", {7, 4});
    run(m, "A-repeat:7+1+3", {7, 1, 3});
    llama_model_free(m);
    llama_backend_free();
}
