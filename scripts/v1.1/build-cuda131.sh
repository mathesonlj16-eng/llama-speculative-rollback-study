#!/usr/bin/env bash
# v1.1.0: clean CUDA builds of stock/paired/prototype at the pin with an explicitly selected toolkit.
# Same CMake options and patches as scripts/build.sh. Usage:
#   CUDA_HOME=/usr/local/cuda-13.1 [UPSTREAM=<local clone or URL>] [CUDA_HEADER_OVERRIDE=<patched crt/math_functions.h>] \
#     bash scripts/v1.1/build-cuda131.sh <new-dir> [variants...]
# CUDA 13.1 headers with GCC 15 need a local two-declaration exception-specification adjustment (rsqrt, rsqrtf)
# in crt/math_functions.h. Supply your own patched copy via CUDA_HEADER_OVERRIDE; it is bind-mounted in a private
# mount namespace for the build only, so the installed toolkit is never modified. NVIDIA headers are not redistributed.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
root=$(realpath -m "$1"); shift
if [[ $# == 0 ]]; then set -- stock paired prototype; fi
cuda=${CUDA_HOME:?set CUDA_HOME to the toolkit, e.g. /usr/local/cuda-13.1}
pin=787a5612fceef602503e5f0ee6133cdc9b8a85c1
mkdir "$root"
"$cuda/bin/nvcc" --version | tail -2; cc --version | head -1; cmake --version | head -1
git init -q "$root/upstream"
git -C "$root/upstream" fetch --depth 1 "${UPSTREAM:-https://github.com/ggml-org/llama.cpp.git}" "$pin"
test "$(git -C "$root/upstream" rev-parse FETCH_HEAD)" = "$pin"
hdr="$cuda/targets/x86_64-linux/include/crt/math_functions.h"
export CUDACXX="$cuda/bin/nvcc" PATH="$cuda/bin:$PATH"
for variant in "$@"; do
    src="$root/src-$variant"
    git -C "$root/upstream" worktree add --detach "$src" FETCH_HEAD
    case "$variant" in paired) patch=paired-checkpoint;; prototype) patch=target-checkpoint-free;; keep) patch=keep-draft;; stock) patch=;;
        *) echo "Unknown variant $variant" >&2; exit 2;; esac
    if [[ -n $patch ]]; then git -C "$src" apply "$repo/patches/$patch.patch"; fi
    configure_and_build="cmake -S '$src' -B '$root/build-$variant' -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && \
        cmake --build '$root/build-$variant' --target llama-server --parallel ${JOBS:-6}"
    if [[ -n ${CUDA_HEADER_OVERRIDE:-} ]]; then
        unshare -Urm bash -ec "mount --bind '$CUDA_HEADER_OVERRIDE' '$hdr'; $configure_and_build"
    else
        bash -ec "$configure_and_build"
    fi
done
