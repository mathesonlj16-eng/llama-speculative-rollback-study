#!/bin/bash
# v1.6.0: fast batch-invariant GEMV kernels for PR #32081 on sm_75 (Q8_0 N=1..8, Q4_K/Q6_K N=1), with the v1.5.0 FA fix.
# Usage: TARGET=/abs/target.gguf DRAFT=/abs/draft.gguf bash scripts/v1.6/run.sh build|sweep|batch|model|perf|bench|verify|spec
# Run scripts/v1.4/run.sh build and e1, and scripts/v1.5/run.sh build, first: they provide pr-bi, pr-bi-fafix,
# the patched replay tools (invariance, logit-dump, logit-cmp) and the reference outputs this script compares against.
# Parameterized copy of the archived drivers (ctrl.sh, kq.sh, kqperf.sh, q8v3/run.sh, nb/run.sh, smalln/run.sh, spec/run.sh);
# it has not been rerun end to end. See provenance/v1.6/FINDINGS.md. sm_75 (Turing) only: the gate disables the kernels elsewhere.
set -euo pipefail
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; Q=$R/qwen-bi; O=$Q/v1.6
REPO=$(cd "$(dirname "$0")/../.." && pwd); S=$REPO/scripts/v1.6
H=3f25ea199a36a9e83480b6dac0941bb82c96c5b9
W0=$R/worktrees/pr-bi; B0=$R/build/pr-bi; W=$R/worktrees/pr-bi-gemv; B=$R/build/pr-bi-gemv
T=$REPO/results/ngram-token-fixtures.tsv; NVCC=/usr/local/cuda-13.2/bin/nvcc
mkdir -p $O
tool() { g++ -O2 -std=c++17 -I$W/include -I$W/ggml/include "$2" -o $O/$1 -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin; }
cu() { # name, source, link build; harness device code comes from the patched tree
  $NVCC -O3 -use_fast_math -arch=sm_75 -std=c++17 -I$W/ggml/src/ggml-cuda -I$W/ggml/include -I$S "$2" -o $O/$1 \
    -L$R/build/$3/bin -lggml -lggml-base -lggml-cuda -Xlinker -rpath=$R/build/$3/bin; }
clocks() { echo "$1,$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader)" >> $O/clocks.csv; }
case ${1:?step} in
build)
  printf '%s  %s\n%s  %s\n' e8bf47d40c5ba68baf8f3e503ee36710da6e8641cb0a670a508cff5b3f0cfbf7 "$TARGET" \
    b4e71cb32694733ba40c69e595a0524dfbc2cabe3c28fb7fef99eed4cf8730bf "$DRAFT" | sha256sum --check
  [ -d $W ] || git -C $R/upstream.git worktree add --detach $W $H
  git -C $W diff --quiet && git -C $W apply $REPO/patches/v1.6/invariant-gemv-fafix-vs-pr-head.patch
  cmake -S $W -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=$NVCC \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
  cmake --build $B --target test-backend-ops llama -j8          # about 12 min cold
  tool invariance $Q/src/invariance.cpp; tool logit-dump $Q/src/logit-dump.cpp   # v1.4-patched sources
  tool bench-bi $REPO/scripts/v1.5/bench-bi.cpp; tool bench-nb $S/bench-nb.cpp; tool spec-bi $S/spec-bi.cpp
  cu ctrl-pr-bi $S/bi_n1_ctrl.cu pr-bi; cu ctrl-gemv $S/bi_n1_ctrl.cu pr-bi-gemv
  cu kq-pr-bi $S/bi_n1_kq.cu pr-bi;     cu kq-gemv $S/bi_n1_kq.cu pr-bi-gemv
  cu variants $S/bi_n1_kq_perf.cu pr-bi ;;
sweep)  # operator bit identity + controls. Linked to pr-bi, every reference N is forced MMQ (tests the harness copies);
        # linked to pr-bi-gemv, N = 1..8 run the in-tree kernels and each column is compared with N = 16 (forced MMQ).
  cd $O
  for L in pr-bi gemv; do rm -f q8-$L.csv kq-$L.csv
    for s in 2048:16:20 2048:512:20 2048:2048:20 2048:4096:10 2048:6144:8 6144:2048:8 4096:4096:8; do IFS=: read K M N <<< "$s"
      ./ctrl-$L $K $M $N 012345 $L >> q8-$L.csv 2>> q8-$L.summary; done
    ./ctrl-$L 2048 248320 1 02 $L >> q8-$L.csv 2>> q8-$L.summary
    for s in q4_K:1024:16:20 q4_K:1024:512:20 q4_K:1024:2048:12 q4_K:1024:3584:8 q4_K:1024:4096:6 q4_K:1024:6144:6 q4_K:2048:1024:12 \
             q4_K:3584:1024:12 q6_K:1024:512:20 q6_K:1024:6144:6 q6_K:3584:1024:12; do IFS=: read t K M N <<< "$s"
      ./kq-$L $t $K $M $N 01234567 $L >> kq-$L.csv 2>> kq-$L.summary; done
    ./kq-$L q6_K 1024 248320 1 02 $L >> kq-$L.csv 2>> kq-$L.summary
  done
  grep -h SUMMARY q8-*.summary kq-*.summary
  # kernel-variant sweep (correctness vs forced MMQ + CUDA-event timing), shapes as archived in results/v1.6/variants-*.csv
  for s in q8_0:2048:16 q8_0:2048:512 q8_0:2048:2048 q8_0:2048:4096 q8_0:2048:6144 q8_0:6144:2048 q8_0:4096:4096 \
           q4_K:1024:512 q4_K:1024:2048 q4_K:1024:3584 q4_K:1024:6144 q4_K:2048:1024 q4_K:3584:1024 q4_K:4096:4096 \
           q6_K:1024:512 q6_K:1024:6144 q6_K:3584:1024 q6_K:4096:4096; do
    IFS=: read t K M <<< "$s"; ./variants $t $K $M 2> /dev/null | tail -n +2; done > variants.csv ;;
batch)
  $B/bin/test-backend-ops batch -o MUL_MAT -b CUDA0 | tail -2
  GGML_CUDA_BI_FAST_N1=0 $B/bin/test-backend-ops batch -o MUL_MAT -b CUDA0 | tail -2 ;;
model)  # 2B target with BI on: E1 vs pr-bi-fafix (v1.5 check output) and the 19-sequence plans; BI off vs PR head default
  mkdir -p $O/e1/on $O/e1/off $O/dumps
  BI=1 $O/invariance "$TARGET" 99 8 cuda-on $O/e1/on > $O/e1-flag-on.csv 2> $O/e1-on.log
  env -u BI $O/invariance "$TARGET" 99 8 cuda-off $O/e1/off > $O/e1-flag-off.csv 2> $O/e1-off.log
  for f in $Q/fa/e1/on/*.f32; do cmp $f $O/e1/on/$(basename $f); done
  for f in $Q/runs/off/*.f32; do cmp $f $O/e1/off/$(basename $f); done
  for s in singles whole c4 c8 pair4; do
    BI=1 $O/logit-dump "$TARGET" 99 $T $s $O/dumps/on-$s.bin 2> $O/dumps/on-$s.log
    cmp $O/dumps/on-$s.bin $Q/fa/dumps/on-$s.bin && echo "$s: identical to pr-bi-fafix"; done
  sha256sum $O/dumps/*.bin ;;   # archived digest of all five plans: results/v1.6/dumps-q8v3-SHA256SUMS.txt
perf)   # matched per-op timing: one binary, arms chosen by env vars only, 5 rounds in shuffled order
  TB=$B/bin/test-backend-ops; mkdir -p $O/perf
  for rep in 1 2 3 4 5; do
    for arm in $(printf "default\nbi-mmq\nbi-v1\nbi-fast\n" | shuf --random-source=<(yes $rep)); do clocks perf-$arm-r$rep
      case $arm in default) e=();; bi-mmq) e=(LLAMA_TEST_BATCH_INVARIANT_PERF=1 GGML_CUDA_BI_FAST_N1=0);;
        bi-v1) e=(LLAMA_TEST_BATCH_INVARIANT_PERF=1 GGML_CUDA_BI_FAST_N1_Q8V3=0);; bi-fast) e=(LLAMA_TEST_BATCH_INVARIANT_PERF=1);; esac
      env "${e[@]}" $TB perf -o MUL_MAT -b CUDA0 -p 'type_a=(q8_0|q4_K|q6_K),type_b=f32,m=[0-9]+,n=[1-8],k=' > $O/perf/$arm-r$rep.log 2>&1
    done; done ;;
bench)  # decode/prefill (bench-bi), target and draft, 5 interleaved rounds
  for m in target draft; do M=$TARGET; [ $m = draft ] && M=$DRAFT; f=$O/bench-$m.csv; echo "label,bi,depth,pp_ms,tg_ms,tg_tokens" > $f
    for rep in 1 2 3 4 5; do for cfg in def bi-fast bi-mmq; do clocks bench-$m-$cfg-r$rep
      case $cfg in def) env -u BI $O/bench-bi "$M" $cfg-r$rep >> $f;; bi-fast) BI=1 $O/bench-bi "$M" $cfg-r$rep >> $f;;
        bi-mmq) BI=1 GGML_CUDA_BI_FAST_N1=0 $O/bench-bi "$M" $cfg-r$rep >> $f;; esac 2>> $O/bench.log; done; done; done ;;
verify) # verify-sized batches (n tokens, logits for all) at depth 1024, 5 interleaved rounds
  f=$O/verify-batches.csv; echo "label,bi,n,reps,ms_total,ms_per_batch,ms_per_token" > $f
  for rep in 1 2 3 4 5; do for cfg in def bi-nb bi-mmq; do clocks verify-$cfg-r$rep
    case $cfg in def) env -u BI $O/bench-nb "$TARGET" $cfg-r$rep >> $f;; bi-nb) BI=1 $O/bench-nb "$TARGET" $cfg-r$rep >> $f;;
      bi-mmq) BI=1 GGML_CUDA_BI_FAST_NB=0 $O/bench-nb "$TARGET" $cfg-r$rep >> $f;; esac 2>> $O/verify.log; done; done
  python3 -I $S/nb_summary.py $O ;;
spec)   # greedy speculative decoding vs plain, tokens + full-logit hashes; 4 arms in round 1, deep context, 2 timing rounds
  mkdir -p $O/spec; cd $O/spec; KS=-3,0,1,2,3,4,5,7
  run() { clocks spec-$1; if [ $2 = 1 ]; then BI=1 TRB=$3 PREFIX=$4 $O/spec-bi "$TARGET" "$DRAFT" $1 $5 $T > $1.csv 2> $1.log
          else env -u BI TRB=$3 PREFIX=$4 $O/spec-bi "$TARGET" "$DRAFT" $1 $5 $T > $1.csv 2> $1.log; fi; }
  run d1000-bi-rs 1 rs 1000 -3,0,3,7; run d1000-def-rs 0 rs 1000 -3,0,3,7
  run r1-bi-rs 1 rs 0 $KS; run r1-def-rs 0 rs 0 $KS; run r1-bi-ckpt 1 ckpt 0 $KS; run r1-def-ckpt 0 ckpt 0 $KS
  for rep in 2 3; do for a in $(printf "bi-rs\ndef-rs\nbi-ckpt\n" | shuf --random-source=<(yes $rep)); do
    case $a in bi-rs) run r$rep-bi-rs 1 rs 0 $KS;; def-rs) run r$rep-def-rs 0 rs 0 $KS;; bi-ckpt) run r$rep-bi-ckpt 1 ckpt 0 $KS;; esac; done; done
  python3 -I $S/compare.py r1-bi-rs.tok r1-bi-ckpt.tok r1-def-rs.tok r1-def-ckpt.tok
  python3 -I $S/compare.py d1000-bi-rs.tok d1000-def-rs.tok
  python3 -I $S/spec_summary.py $O/spec ;;
*) echo "unknown step $1" >&2; exit 2 ;;
esac
