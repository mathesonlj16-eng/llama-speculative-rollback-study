// Mutation for the oracle: every second partial tail removal reports success without removing anything,
// leaving rejected tokens in the recurrent state. The oracle must detect this.
#include "llama.h"
#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<long> g_calls{0}, g_skipped{0};

bool llama_memory_seq_rm(llama_memory_t m, llama_seq_id s, llama_pos p0, llama_pos p1) {
    static auto real = (decltype(&llama_memory_seq_rm)) dlsym(RTLD_NEXT, "llama_memory_seq_rm");
    if (p0 > 0 && p1 < 0 && (++g_calls % 2) == 0) {
        ++g_skipped;
        return true;
    }
    return real(m, s, p0, p1);
}
__attribute__((destructor)) static void report() {
    fprintf(stderr, "BREAK_ROLLBACK partial_calls=%ld skipped=%ld\n", g_calls.load(), g_skipped.load());
}
