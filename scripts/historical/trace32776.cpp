#include "llama.h"
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

class trace_range {
public:
    trace_range(const char * operation, llama_context * ctx, unsigned flags, size_t bytes) {
        char name[128];
        std::snprintf(name, sizeof(name), "%s %p %u %zu", operation, static_cast<void *>(ctx), flags, bytes);
        nvtxRangePushA(name);
    }
    ~trace_range() { nvtxRangePop(); }
};

llama_context * llama_init_from_model(llama_model * model, llama_context_params params) {
    static const auto real = reinterpret_cast<decltype(&llama_init_from_model)>(dlsym(RTLD_NEXT, "llama_init_from_model"));
    if (std::getenv("LLAMA32776_TARGET_ROLLBACK") && llama_model_n_params(model) > 1500000000ULL) {
        params.n_rs_seq = 3;
        std::fprintf(stderr, "local experiment: target n_rs_seq = %u\n", params.n_rs_seq);
    }
    return real(model, params);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags) {
    static const auto real = reinterpret_cast<decltype(&llama_state_seq_get_size_ext)>(dlsym(RTLD_NEXT, "llama_state_seq_get_size_ext"));
    const trace_range range("state_size", ctx, flags, 0);
    return real(ctx, seq, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq, llama_state_seq_flags flags) {
    static const auto real = reinterpret_cast<decltype(&llama_state_seq_get_data_ext)>(dlsym(RTLD_NEXT, "llama_state_seq_get_data_ext"));
    const trace_range range("state_save", ctx, flags, size);
    return real(ctx, dst, size, seq, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq, llama_state_seq_flags flags) {
    static const auto real = reinterpret_cast<decltype(&llama_state_seq_set_data_ext)>(dlsym(RTLD_NEXT, "llama_state_seq_set_data_ext"));
    const trace_range range("state_restore", ctx, flags, size);
    return real(ctx, src, size, seq, flags);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    static const auto real = reinterpret_cast<decltype(&llama_process)>(dlsym(RTLD_NEXT, "llama_process"));
    const trace_range range("process", ctx, type, 0);
    return real(ctx, type, batch);
}
