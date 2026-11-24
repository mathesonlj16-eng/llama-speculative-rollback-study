// Minimal preload for the comparison server: sets target rollback capacity and does nothing else.
#include "llama.h"
#include <cstdlib>
#include <dlfcn.h>

llama_context * llama_init_from_model(llama_model * m, llama_context_params p) {
    static auto real = (decltype(&llama_init_from_model)) dlsym(RTLD_NEXT, "llama_init_from_model");
    if (llama_model_n_params(m) > 1500000000ULL) {
        if (const char * v = std::getenv("Q35_TARGET_RS")) p.n_rs_seq = (uint32_t) std::atoi(v);
    }
    return real(m, p);
}
