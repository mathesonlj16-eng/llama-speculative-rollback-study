// Timing for batch-invariant FA cost: prefill d tokens (ubatch 512), then 64 one-token decodes at depth d.
// usage: model label ; env BI=1 enables batch_invariant. Prints CSV rows: label,bi,depth,pp_ms,tg_ms,tg_tokens
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void decode(llama_context * c, int pos0, int n, bool want_logits) {
    auto * b = llama_batch_ext_init(c);
    for (int j = 0; j < n; j++) { const int p = pos0 + j;
        int i = llama_batch_ext_add_token(b, 0, 1000 + (p * 7919) % 20000);
        if (i < 0 || !llama_batch_ext_set_pos(b, i, &p) || !llama_batch_ext_set_output_logits(b, i, want_logits && j == n - 1)) throw std::runtime_error("batch"); }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b)) throw std::runtime_error("decode");
    if (want_logits) { if (!llama_get_logits_ith(c, n - 1)) throw std::runtime_error("logits"); }
    llama_batch_ext_free(b);
}
int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: model label\n"); return 2; }
    const bool bi = getenv("BI") != nullptr;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * m = llama_model_load_from_file(argv[1], mp); if (!m) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4608; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = 8; cp.batch_invariant = bi;
    auto * c = llama_init_from_model(m, cp); if (!c) return 1;
    const int TG = 64;
    for (int depth : {256, 256, 1024, 4096}) {   // first 256 is a warm-up, reported as depth -256
        static bool first = true;
        llama_memory_clear(llama_get_memory(c), true); llama_synchronize(c);
        double t0 = now_ms();
        for (int p = 0; p < depth; p += 512) decode(c, p, std::min(512, depth - p), p + 512 >= depth);
        llama_synchronize(c); double t1 = now_ms();
        for (int k = 0; k < TG; k++) decode(c, depth + k, 1, true);
        llama_synchronize(c); double t2 = now_ms();
        printf("%s,%d,%d,%.3f,%.3f,%d\n", argv[2], bi, first ? -depth : depth, t1 - t0, t2 - t1, TG); fflush(stdout);
        first = false;
    }
    llama_free(c); llama_model_free(m); llama_backend_free();
}
