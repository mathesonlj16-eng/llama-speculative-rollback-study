#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=$(realpath -m "$1")
backend=$2
shift 2
if [[ $# == 0 ]]; then set -- stock paired prototype keep exact; fi
case "$backend" in
    cpu) options=(-DGGML_CUDA=OFF);;
    cuda) options=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75");;
    *) echo "Expected cpu or cuda" >&2; exit 2;;
esac
mkdir "$root"
pin=787a5612fceef602503e5f0ee6133cdc9b8a85c1
git init -q "$root/upstream"
git -C "$root/upstream" fetch --depth 1 https://github.com/ggml-org/llama.cpp.git "$pin"
for variant in "$@"; do
    if [[ $backend == cuda && $variant == exact ]]; then continue; fi
    src="$root/src-$variant"
    git -C "$root/upstream" worktree add --detach "$src" FETCH_HEAD
    case "$variant" in
        paired) patch=paired-checkpoint;;
        prototype) patch=target-checkpoint-free;;
        keep) patch=keep-draft;;
        exact) patch=exact-activation;;
        stock) patch=;;
        *) echo "Unknown variant $variant" >&2; exit 2;;
    esac
    if [[ -n $patch ]]; then git -C "$src" apply "$repo/patches/$patch.patch"; fi
    cmake -S "$src" -B "$root/build-$variant" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF "${options[@]}"
    cmake --build "$root/build-$variant" --target llama-server --parallel "${JOBS:-6}"
done
