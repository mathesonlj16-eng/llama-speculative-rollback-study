// Timing preload. Logs one line per outermost llama API call:
//   role op n_tokens start_ns duration_ns thread
// role: T target, D draft. op: P process, S synchronize, L get_logits, G state get, R state set, M seq_rm.
// Also applies Q35_TARGET_RS to the target context, as cap.cpp does.
#include "llama-batch.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <pthread.h>

static std::mutex g_lock;
static FILE * g_out = nullptr;
static std::map<const void *, char> g_roles;
static thread_local int g_depth = 0;

static long long now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}
static char role(const void * key) {
    std::lock_guard<std::mutex> l(g_lock);
    auto it = g_roles.find(key);
    return it == g_roles.end() ? '?' : it->second;
}
static void emit(char r, char op, size_t n, long long start) {
    const long long d = now() - start;
    std::lock_guard<std::mutex> l(g_lock);
    if (!g_out) {
        const char * p = std::getenv("Q35_PROF");
        if (!p) return;
        g_out = std::fopen(p, "w");
        if (!g_out) return;
    }
    std::fprintf(g_out, "%c %c %zu %lld %lld %lu\n", r, op, n, start, d, (unsigned long) pthread_self());
    std::fflush(g_out);
}
template <typename F> static auto timed(char r, char op, size_t n, F && f) {
    if (g_depth > 0) return f();
    ++g_depth;
    const long long t = now();
    auto result = f();
    emit(r, op, n, t);
    --g_depth;
    return result;
}

llama_context * llama_init_from_model(llama_model * m, llama_context_params p) {
    static auto real = (decltype(&llama_init_from_model)) dlsym(RTLD_NEXT, "llama_init_from_model");
    const bool target = llama_model_n_params(m) > 1500000000ULL;
    if (target) {
        if (const char * v = std::getenv("Q35_TARGET_RS")) p.n_rs_seq = (uint32_t) std::atoi(v);
    }
    auto * c = real(m, p);
    if (c) {
        std::lock_guard<std::mutex> l(g_lock);
        g_roles[c] = target ? 'T' : 'D';
        g_roles[llama_get_memory(c)] = target ? 'T' : 'D';
    }
    return c;
}
int32_t llama_process(llama_context * c, llama_process_type t, llama_batch_ext * b) {
    static auto real = (decltype(&llama_process)) dlsym(RTLD_NEXT, "llama_process");
    return timed(role(c), 'P', b->tokens.size(), [&] { return real(c, t, b); });
}
void llama_synchronize(llama_context * c) {
    static auto real = (decltype(&llama_synchronize)) dlsym(RTLD_NEXT, "llama_synchronize");
    timed(role(c), 'S', 0, [&] { real(c); return 0; });
}
float * llama_get_logits_ith(llama_context * c, int32_t i) {
    static auto real = (decltype(&llama_get_logits_ith)) dlsym(RTLD_NEXT, "llama_get_logits_ith");
    return timed(role(c), 'L', 0, [&] { return real(c, i); });
}
size_t llama_state_seq_get_data_ext(llama_context * c, uint8_t * d, size_t n, llama_seq_id s, llama_state_seq_flags f) {
    static auto real = (decltype(&llama_state_seq_get_data_ext)) dlsym(RTLD_NEXT, "llama_state_seq_get_data_ext");
    return timed(role(c), 'G', n, [&] { return real(c, d, n, s, f); });
}
size_t llama_state_seq_set_data_ext(llama_context * c, const uint8_t * d, size_t n, llama_seq_id s, llama_state_seq_flags f) {
    static auto real = (decltype(&llama_state_seq_set_data_ext)) dlsym(RTLD_NEXT, "llama_state_seq_set_data_ext");
    return timed(role(c), 'R', n, [&] { return real(c, d, n, s, f); });
}
bool llama_memory_seq_rm(llama_memory_t m, llama_seq_id s, llama_pos p0, llama_pos p1) {
    static auto real = (decltype(&llama_memory_seq_rm)) dlsym(RTLD_NEXT, "llama_memory_seq_rm");
    return timed(role(m), 'M', 0, [&] { return real(m, s, p0, p1); });
}
