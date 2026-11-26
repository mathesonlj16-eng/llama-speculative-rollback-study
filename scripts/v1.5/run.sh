#!/bin/bash
# v1.5.0: invariant-mode flash-attention KV-split fix on PR #32081 head: correctness and cost.
# Usage: TARGET=/path/to/target.gguf bash scripts/v1.5/run.sh build|check|bench|padkv
# Run `bash scripts/v1.4/run.sh build` first: it provides the unpatched build, the patched replay tools and logit-cmp.
# Parameterized copy of the archived drivers (driver-fa.sh, driver-bench.sh, driver-padkv.sh); see provenance/v1.5/FINDINGS.md.
set -euo pipefail
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; Q=$R/qwen-bi; A=$Q/fa
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H=3f25ea199a36a9e83480b6dac0941bb82c96c5b9
W0=$R/worktrees/pr-bi; B0=$R/build/pr-bi; W=$R/worktrees/pr-bi-fafix; B=$R/build/pr-bi-fafix
T=$REPO/results/ngram-token-fixtures.tsv
tool() { # name, source, worktree, build
  g++ -O2 -std=c++17 -I$3/include -I$3/ggml/include "$2" -o $Q/$1 -L$4/bin -lllama -lggml -lggml-base -Wl,-rpath,$4/bin; }
case ${1:?step} in
build)
  printf '%s  %s\n' e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7 "$TARGET" | sha256sum --check
  [ -d $W ] || git -C $R/upstream.git worktree add --detach $W $H
  git -C $W diff --quiet && git -C $W apply $REPO/patches/v1.5/fa-kv-split-invariant-vs-pr-head.patch
  cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
  cmake --build $B --target llama -j8
  tool invariance-fafix $Q/src/invariance.cpp $W $B
  tool logit-dump-fafix $Q/src/logit-dump.cpp $W $B
  cp $REPO/scripts/v1.5/bench-bi.cpp $Q/src/
  tool bench-pr-bi $Q/src/bench-bi.cpp $W0 $B0
  tool bench-pr-bi-fafix $Q/src/bench-bi.cpp $W $B ;;
check)  # E1 off/on; teacher-forced shapes flag on (no n_kv pin); default mode vs the unpatched build
  mkdir -p $A/e1/off $A/e1/on $A/dumps $A/cmp
  for arm in off on; do if [ $arm = on ]; then export BI=1; else unset BI; fi
    $Q/invariance-fafix "$TARGET" 99 8 cuda-$arm $A/e1/$arm > $A/e1-$arm.csv 2> $A/e1-$arm.log; done
  unset BI
  for f in $Q/runs/off/*.f32; do cmp $f $A/e1/off/$(basename $f); done   # needs v1.4 `run.sh e1` output
  export BI=1
  for s in singles whole c4 c8 pair4; do $Q/logit-dump-fafix "$TARGET" 99 $T $s $A/dumps/on-$s.bin 2> $A/dumps/on-$s.log; done
  unset BI
  for s in whole c4 c8 pair4; do $Q/logit-cmp $T $A/dumps/on-singles.bin $A/dumps/on-$s.bin on-singles-vs-$s > $A/cmp/on-$s.csv; done
  for s in singles whole; do
    $Q/logit-dump-fafix "$TARGET" 99 $T $s $A/dumps/off-fafix-$s.bin 2> $A/dumps/off-fafix-$s.log
    $Q/logit-dump "$TARGET" 99 $T $s $A/dumps/off-base-$s.bin 2> $A/dumps/off-base-$s.log
    $Q/logit-cmp $T $A/dumps/off-base-$s.bin $A/dumps/off-fafix-$s.bin default-base-vs-fafix-$s > $A/cmp/default-$s.csv; done ;;
bench)  # interleaved: unpatched/patched x default/invariant, 5 reps
  mkdir -p $Q/bench; echo "label,bi,depth,pp_ms,tg_ms,tg_tokens" > $Q/bench/times.csv
  for rep in 1 2 3 4 5; do for cfg in base-def base-bi fafix-bi fafix-def; do
    case $cfg in base-*) b=$Q/bench-pr-bi;; *) b=$Q/bench-pr-bi-fafix;; esac
    case $cfg in *-bi) export BI=1;; *) unset BI;; esac
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader >> $Q/bench/gpu.txt
    $b "$TARGET" $cfg-r$rep >> $Q/bench/times.csv 2> $Q/bench/$cfg-r$rep.log; done; done
  unset BI
  python3 -I $REPO/scripts/v1.5/bench_an.py $Q/bench/times.csv ;;
padkv)  # v1.5.1: operator-level FA test with llama.cpp's padded K/V length, against both builds (no model)
  P=$(mktemp -d); mkdir -p $P/tests $Q/padkv; cp $W0/tests/test-backend-ops.cpp $P/tests/
  patch -p1 -d $P < $REPO/patches/v1.5/test-backend-ops-fa-padkv-vs-pr-head.patch
  c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -I$W0/ggml/include -O3 -DNDEBUG -std=c++17     -c $P/tests/test-backend-ops.cpp -o $Q/padkv/tbo.o
  for v in pr-bi pr-bi-fafix; do b=$R/build/$v/bin
    c++ -O3 $Q/padkv/tbo.o -o $Q/padkv/tbo-$v -Wl,-rpath,$b $b/libggml.so.1.0.0 $b/libggml-cpu.so.1.0.0 $b/libggml-cuda.so.1.0.0       $b/libggml-base.so.1.0.0 /usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs/libcuda.so
    $Q/padkv/tbo-$v batch -o FLASH_ATTN_EXT -b CUDA0 > $Q/padkv/run-$v.txt 2>&1 && echo "$v: pass" || echo "$v: FAIL (exit $?)"
  done ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
