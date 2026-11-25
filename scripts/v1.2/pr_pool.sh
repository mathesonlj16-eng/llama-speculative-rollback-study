#!/usr/bin/env bash
# PR large-N correctness: new worktree = audited pr diff + dp4a switch (macro-guarded) + the same 5 context-pool
# enlargements the fixed-warp harness got. Two builds: with switch (as d4f-pr) and without (as the audited sm_75 pr).
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; G=$D/upstream.git; W=$D/worktrees/pr-pool; X=$E/pr-pool
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
mkdir -p $X; L=$X/log.txt; st() { echo "$(date -Is) $*" | tee -a $L; }
[ -e $W ] && { st "worktree exists, refusing"; exit 1; }
git -C $G worktree add --detach $W $(git -C $D/worktrees/pr rev-parse HEAD) > /dev/null 2>&1
git -C $W apply $E/worktree-dp4a-pr.full.diff || { st "apply failed"; exit 1; }
T=$W/tests/test-backend-ops.cpp
sed -i -e 's|ggml_tensor_overhead()\*256 + ggml_graph_overhead(),|ggml_tensor_overhead()*4096 + ggml_graph_overhead(), // SM75 local patch: room for large-N batch tests|' \
       -e 's|ggml_tensor_overhead()\*128 + ggml_graph_overhead_custom(graph_nodes, false),|ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(graph_nodes, false), // SM75 local patch: room for large-N batch tests|' \
       -e 's|ggml_tensor_overhead()\*128 + ggml_graph_overhead(),|ggml_tensor_overhead()*4096 + ggml_graph_overhead(), // SM75 local patch: room for large-N batch tests|' $T
n=$(grep -c 'room for large-N batch tests' $T); st "pool edits applied: $n (expect 5)"; [ "$n" = 5 ] || exit 1
git -C $W diff > $X/worktree-pr-pool.full.diff
build() { local name=$1 force=$2 flags=$3 B=$D/build/$1 t0=$(date +%s)
  [ -e $B ] && { st "build dir $B exists"; return 1; }
  cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_FORCE_MMQ=$force \
    "-DCMAKE_CUDA_FLAGS=$flags" "-DCMAKE_CXX_FLAGS=$flags" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > $X/build-$name.configure.log 2>&1 || { st "configure FAIL $name"; return 1; }
  cmake --build $B --target test-backend-ops -j8 > $X/build-$name.log 2>&1 || { st "build FAIL $name"; return 1; }
  st "built $name in $(( $(date +%s)-t0 )) s"; }
build pool-d4f-pr ON -DGGML_CUDA_DIAG_NO_TURING_MMA || exit 1
build pool-s75-pr OFF "" || exit 1
export SM75_BATCH_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_BATCH_M=256,1024 SM75_BATCH_K=1024,4096,5120 SM75_BATCH_N=9,10,11,12,13,14,15,16,24,32,48,64,128,256,512 SM75_HASH=1
env | grep ^SM75_ > $X/env.txt
for seed in 1234 rand; do for b in pool-d4f-pr pool-s75-pr; do f=$X/$b.$seed.log
  if [ $seed = rand ]; then unset SM75_SEED; else export SM75_SEED=$seed; fi
  echo "build=$b seed=$seed $(date -Is)" > $f; $D/build/$b/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT >> $f 2>&1; echo "exit=$?" >> $f
  st "$b $seed: $(tail -1 $f) ok=$(grep -c 'MUL_MAT.*OK' $f) fail=$(grep -c 'MUL_MAT.*FAIL' $f)"; done; done
st DONE
