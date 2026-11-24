// Second mutation: every second partial-state restore reports success without restoring anything,
// so checkpoint-based rollback keeps the rejected tokens. The oracle must detect this.
#include "llama.h"
#include <atomic>
#include <cstdio>
#include <dlfcn.h>

static std::atomic<long> g_calls{0}, g_skipped{0};

size_t llama_state_seq_set_data_ext(llama_context * c, const uint8_t * d, size_t n, llama_seq_id s, llama_state_seq_flags f) {
    static auto real = (decltype(&llama_state_seq_set_data_ext)) dlsym(RTLD_NEXT, "llama_state_seq_set_data_ext");
    if ((f & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && (++g_calls % 2) == 0) {
        ++g_skipped;
        return n;
    }
    return real(c, d, n, s, f);
}
__attribute__((destructor)) static void report() {
    fprintf(stderr, "BREAK_RESTORE partial_restores=%ld skipped=%ld\n", g_calls.load(), g_skipped.load());
}
