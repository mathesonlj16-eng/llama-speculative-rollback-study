// G6 preload: turns on the CPU backend's reference mode, which disables tiled and llamafile matrix kernels.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <dlfcn.h>

ggml_backend_t ggml_backend_cpu_init(void) {
    static auto real = (ggml_backend_t (*)(void)) dlsym(RTLD_NEXT, "ggml_backend_cpu_init");
    ggml_backend_t b = real();
    if (b) {
        ggml_backend_cpu_set_use_ref(b, true);
        fprintf(stderr, "USEREF applied\n");
    }
    return b;
}
