#!/usr/bin/env bash
# dp4a-switch builds for the GTX 1650: same toolkit (CUDA 13.2), same sm_75 target as the audited builds,
# plus -DGGML_CUDA_DIAG_NO_TURING_MMA. Sequential; only target test-backend-ops. Existing builds untouched.
#   d4f-fw : dp4a-fw worktree, GGML_CUDA_FORCE_MMQ=ON
#   d4f-pr : dp4a-pr worktree, GGML_CUDA_FORCE_MMQ=ON
#   d4-fw  : dp4a-fw worktree, GGML_CUDA_FORCE_MMQ=OFF (default dispatch -> cuBLAS at N>=64)
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; T=$E/build-times.txt
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
NVCC=/usr/local/cuda-13.2/bin/nvcc
build() { local name=$1 wt=$2 force=$3; local B=$D/build/$name W=$D/worktrees/$wt
  [ -e $B ] && { echo "build dir $B exists, refusing" >> $T; return 1; }
  echo "=== $name start $(date -Is) wt=$wt force=$force" >> $T
  ccache -z >/dev/null; local t0=$(date +%s)
  cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_FORCE_MMQ=$force \
    -DCMAKE_CUDA_FLAGS=-DGGML_CUDA_DIAG_NO_TURING_MMA -DCMAKE_CXX_FLAGS=-DGGML_CUDA_DIAG_NO_TURING_MMA \
    -DCMAKE_CUDA_COMPILER=$NVCC -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > $E/build-$name.configure.log 2>&1 || { echo "configure FAIL $name" >> $T; return 1; }
  cmake --build $B --target test-backend-ops -j8 > $E/build-$name.log 2>&1 || { echo "build FAIL $name" >> $T; return 1; }
  echo "=== $name done $(date -Is) elapsed_s=$(( $(date +%s)-t0 ))" >> $T
  ccache -s | grep -E "Hits|Misses" >> $T
  # Flags nvcc actually received for the MMQ translation units
  grep -o '"command": "[^"]*mmq[^"]*"' $B/compile_commands.json | grep -oE -- '--generate-code=[^ ]*|-gencode[= ][^ ]*|-arch[= ][^ ]*|-DGGML_CUDA_FORCE_MMQ|-DGGML_CUDA_DIAG_NO_TURING_MMA' | sort | uniq -c > $E/nvcc-flags-$name.txt
}
build d4f-fw dp4a-fw ON  || exit 1
build d4f-pr dp4a-pr ON  || exit 1
build d4-fw  dp4a-fw OFF || exit 1
echo ALLDONE >> $T
