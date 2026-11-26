// Small-batch ("verify") timing: prefill DEPTH tokens, then for each n run REPS batches of n tokens
// (logits for every token, as speculative-decoding verification needs), advancing positions.
// usage: model label ; env BI=1 enables batch_invariant. CSV: label,bi,n,reps,ms_total,ms_per_batch,ms_per_token
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void decode(llama_context * c, int pos0, int n, bool all_logits) {
    auto * b = llama_batch_ext_init(c);
    for (int j = 0; j < n; j++) { const int p = pos0 + j;
        int i = llama_batch_ext_add_token(b, 0, 1000 + (p * 7919) % 20000);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &p) || !llama_batch_ext_set_output_logits(b, i, all_logits || j == n - 1)) throw std::runtime_error("batch"); }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    if (!llama_get_logits_ith(c, n - 1)) throw std::runtime_error("logits");
    llama_batch_ext_free(b);
}
int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: model label\n"); return 2; }
    const bool bi = getenv("BI") != nullptr;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * m = llama_model_load_from_file(argv[1], mp); if (!m) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = 8; cp.batch_invariant = bi;
    auto * c = llama_init_from_model(m, cp); if (!c) return 1;
    const int DEPTH = 1024, REPS = 32;
    for (int n : {4, 1, 2, 3, 4, 5, 6, 8, 12, 16}) {   // first n=4 is a warm-up, reported as n=-4
        static bool first = true;
        llama_memory_clear(llama_get_memory(c), true);
        for (int p = 0; p < DEPTH; p += 512) decode(c, p, std::min(512, DEPTH - p), false);
        llama_synchronize(c); const double t0 = now_ms();
        for (int r = 0; r < REPS; r++) decode(c, DEPTH + r*n, n, true);
        llama_synchronize(c); const double t = now_ms() - t0;
        printf("%s,%d,%d,%d,%.3f,%.4f,%.4f\n", argv[2], bi, first ? -n : n, REPS, t, t/REPS, t/(REPS*n)); fflush(stdout);
        first = false;
    }
    llama_free(c); llama_model_free(m); llama_backend_free();
}
