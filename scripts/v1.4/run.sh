#!/bin/bash
# v1.4.0: batch-invariant replay of the study's grouping fixtures on PR #32081 head.
# Usage: TARGET=/path/to/target.gguf bash scripts/v1.4/run.sh build|e1|a3|a4|a5
# Parameterized copy of the archived drivers (driver.sh, driver-a3.sh, driver-a4.sh, driver-a5.sh); see provenance/v1.4/FINDINGS.md.
# Expects SM75_ROOT (default $HOME/research/sm75-batch-invariance) to hold a bare clone at upstream.git with pull/32081/head fetched.
set -euo pipefail
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; Q=$R/qwen-bi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H=3f25ea199a36a9e83480b6dac0941bb82c96c5b9
W=$R/worktrees/pr-bi; B=$R/build/pr-bi; WK=$R/worktrees/pr-bi-kvfull; BK=$R/build/pr-bi-kvfull
T=$REPO/results/ngram-token-fixtures.tsv
mkdir -p $Q/src
tool() { # name, source, libdir
  g++ -O2 -std=c++17 -I$W/include -I$W/ggml/include "$2" -o $Q/$1 -L$3 -lllama -lggml -lggml-base -Wl,-rpath,$3; }
case ${1:?step} in
build)
  printf '%s  %s\n' e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7 "$TARGET" | sha256sum --check
  [ -d $W ] || git -C $R/upstream.git worktree add --detach $W $H
  cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
  cmake --build $B --target llama -j8
  P=$(mktemp -d); mkdir -p $P/scripts/archive/fixed-token-replay $P/scripts/v1.1
  cp $REPO/scripts/archive/fixed-token-replay/invariance.cpp $P/scripts/archive/fixed-token-replay/
  cp $REPO/scripts/v1.1/logit-dump.cpp $P/scripts/v1.1/
  patch -p1 -d $P < $REPO/patches/v1.4/replay-tools-batch-invariant-toggle.patch
  cp $P/scripts/archive/fixed-token-replay/invariance.cpp $P/scripts/v1.1/logit-dump.cpp $REPO/scripts/v1.1/logit-cmp.cpp $Q/src/
  tool invariance $Q/src/invariance.cpp $B/bin
  tool logit-dump $Q/src/logit-dump.cpp $B/bin
  g++ -O2 -std=c++17 $Q/src/logit-cmp.cpp -o $Q/logit-cmp
  # A4 library: same tree with n_kv pinned to the cache size; only llama-kv-cache.cpp is recompiled, then libllama is relinked.
  [ -d $WK ] || git -C $R/upstream.git worktree add --detach $WK $H
  git -C $WK diff --quiet && git -C $WK apply $REPO/patches/v1.4/kv-n_kv-pinned-vs-pr-head.patch
  rm -rf $BK; cp -a $B $BK
  CMD=$(python3 -I -c 'import json,sys; print(next(e["command"] for e in json.load(open(sys.argv[1])) if e["file"].endswith("/src/llama-kv-cache.cpp")))' $B/compile_commands.json)
  CMD=${CMD//$W\/src\/llama-kv-cache.cpp/$WK\/src\/llama-kv-cache.cpp}; CMD=${CMD//$B\//$BK\/}
  (cd $BK/src && eval "$CMD" && eval "$(cat CMakeFiles/llama.dir/link.txt)")
  tool logit-dump-kvfull $Q/src/logit-dump.cpp $BK/bin ;;
e1)   # A0 (flag off) and A1 (flag on), same binary
  for arm in off on; do mkdir -p $Q/runs/$arm
    if [ $arm = on ]; then export BI=1; else unset BI; fi
    $Q/invariance "$TARGET" 99 8 cuda-$arm $Q/runs/$arm > $Q/runs/e1-$arm.csv 2> $Q/runs/e1-$arm.log; done ;;
a3)   # teacher-forced shapes, flag off and on, each against singles of the same arm
  mkdir -p $Q/a3/dumps $Q/a3/cmp
  for arm in off on; do if [ $arm = on ]; then export BI=1; else unset BI; fi
    for s in singles whole c4 c8 pair4; do $Q/logit-dump "$TARGET" 99 $T $s $Q/a3/dumps/$arm-$s.bin 2> $Q/a3/dumps/$arm-$s.log; done; done
  for arm in off on; do for s in whole c4 c8 pair4; do
    $Q/logit-cmp $T $Q/a3/dumps/$arm-singles.bin $Q/a3/dumps/$arm-$s.bin $arm-singles-vs-$s > $Q/a3/cmp/$arm-$s.csv; done; done
  $Q/logit-cmp $T $Q/a3/dumps/off-singles.bin $Q/a3/dumps/on-singles.bin off-vs-on-singles > $Q/a3/cmp/cross-singles.csv ;;
a4)   # flag on, n_kv pinned
  mkdir -p $Q/a4/dumps $Q/a4/cmp; export BI=1
  for s in singles whole c4 c8 pair4; do $Q/logit-dump-kvfull "$TARGET" 99 $T $s $Q/a4/dumps/on-$s.bin 2> $Q/a4/dumps/on-$s.log; done
  for s in whole c4 c8 pair4; do $Q/logit-cmp $T $Q/a4/dumps/on-singles.bin $Q/a4/dumps/on-$s.bin on-singles-vs-$s > $Q/a4/cmp/on-$s.csv; done
  $Q/logit-cmp $T $Q/a3/dumps/on-singles.bin $Q/a4/dumps/on-singles.bin a3on-vs-a4on-singles > $Q/a4/cmp/cross-singles.csv ;;
a5)   # pair4 control: singles with 1024 cells, as in each pair4 stream
  mkdir -p $Q/a5/dumps $Q/a5/cmp
  BI=1 N_CTX=1024 $Q/logit-dump-kvfull "$TARGET" 99 $T singles $Q/a5/dumps/on-singles-ctx1024.bin 2> $Q/a5/dumps/on-singles-ctx1024.log
  $Q/logit-cmp $T $Q/a5/dumps/on-singles-ctx1024.bin $Q/a4/dumps/on-pair4.bin ctx1024-singles-vs-pair4 > $Q/a5/cmp/ctx1024-vs-pair4.csv
  $Q/logit-cmp $T $Q/a4/dumps/on-singles.bin $Q/a5/dumps/on-singles-ctx1024.bin ctx2048-vs-ctx1024-singles > $Q/a5/cmp/ctx2048-vs-ctx1024.csv ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
