#!/bin/bash
# Archived driver, as run on 2026-11-26 (see REPRODUCE.md). Only the paths were changed for publication:
# SM75_ROOT = research root (worktrees/, build/, models/, qwen-bi/); S = this repository's scripts/ folder.
# Rerun with hybrid Q4_K staging (float4 at NC=2, compact at NC>=3). Q4_K/Q6_K N=2..8 fast path (bi_nb_q4_K_v3 / bi_nb_q6_K_v3) in mmq-n1-fa: build, bit-identity gates, sanitizer,
# SASS op mix, then matched per-op timing and BI speculative decoding (new kernel vs GGML_CUDA_BI_FAST_NB_KQ=0).
# Checks stop the job before any timing if anything differs. Sequential; no compile overlaps timing.
set -u
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; Q=$R/qwen-bi
S=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=$R/worktrees/mmq-n1-fa; B=$R/build/mmq-n1-fa
MT=$R/models/Qwen3.5-2B-Q8_0.gguf; MD=$R/models/Qwen3.5-0.8B-Q4_K_M.gguf; T=$Q/src/ngram-token-fixtures.tsv
SPEC_OLD=$R/scratch-mmq-n1/spec/full      # v1.6.0 speculative run (published as results/v1.6/spec/)
O=$R/scratch-mmq-n1/nbkq3
mkdir -p $O/dumps $O/cmp $O/e1 $O/ctrl $O/spec
log() { echo "$(date -Is) $*" >> $O/timeline.txt; }
fail() { log "FAIL $*"; touch $O/FAIL; exit 1; }
rm -f $O/DONE $O/FAIL $O/checks.txt
export PATH=/usr/local/cuda-13.2/bin:$PATH
log "build start"
cmake --build $B --target test-backend-ops llama -j "$(nproc)" > $O/build.log 2>&1 || fail build
TB=$B/bin/test-backend-ops
for p in invariance logit-dump-nctx; do
  g++ -O2 -std=c++17 -I$W/include -I$W/ggml/include $Q/src/$p.cpp -o $O/$p -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin >> $O/compile.log 2>&1 || fail compile-$p
done
g++ -O2 -std=c++17 -I$W/include -I$W/ggml/include $S/v1.6/spec-bi.cpp -o $O/spec-bi -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin >> $O/compile.log 2>&1 || fail compile-spec
cp $S/v1.7/bi_nb_kq_lib.cu $O/ctrl/
nvcc -O2 -arch=sm_75 -std=c++17 -I$W/ggml/include $O/ctrl/bi_nb_kq_lib.cu -o $O/ctrl/lib \
    -L$B/bin -lggml -lggml-base -lggml-cuda -Xlinker -rpath=$B/bin > $O/ctrl/build.log 2>&1 || fail ctrl-build
log compiled
L=$O/ctrl/lib
# 0. smoke
$L q4_K 1024 512 1 0 smoke >> $O/ctrl/smoke.csv 2>> $O/ctrl/ctrl.summary || fail smoke-q4
$L q6_K 1024 512 1 0 smoke >> $O/ctrl/smoke.csv 2>> $O/ctrl/ctrl.summary || fail smoke-q6
awk -F, '$1=="smoke" && $7>0{b=1} END{exit b}' $O/ctrl/smoke.csv || fail smoke-bits
log smoke-ok
# 1. adversarial sweep on the draft's shapes: every column of N = 1..8 vs N = 16 forced MMQ; default mode as control
for s in q4_K:1024:16:8 q4_K:1024:512:8 q4_K:1024:2048:6 q4_K:1024:3584:4 q4_K:1024:4096:4 q4_K:1024:6144:3 q4_K:2048:1024:6 q4_K:3584:1024:6 \
         q6_K:1024:512:8 q6_K:1024:6144:3 q6_K:3584:1024:6 q6_K:4096:4096:2; do
  IFS=: read ty K MM NS <<< "$s"
  $L $ty $K $MM $NS 0123456 sweep >> $O/ctrl/ctrl.csv 2>> $O/ctrl/ctrl.summary || fail ctrl-$s
done
$L q6_K 1024 248320 1 02 sweep >> $O/ctrl/ctrl.csv 2>> $O/ctrl/ctrl.summary || fail ctrl-head
BAD=$(awk -F, '$1=="sweep"{s+=$7; c+=$8} END{print s+0" of "c+0}' $O/ctrl/ctrl.csv)
CTL=$(awk -F, '$1=="sweep"{n++; if($9>0)f++; s+=$9; c+=$10} END{print "rows failing "f+0" of "n+0", floats "s+0" of "c+0}' $O/ctrl/ctrl.csv)
NZ=$(awk -F, '$1=="sweep"{z+=$11; f+=$12} END{print "ref -0.0 outputs "z+0", non-finite "f+0}' $O/ctrl/ctrl.csv)
echo "sweep: BI cols N=1..8 vs N=16 forced MMQ: bad $BAD; default-mode control: $CTL; $NZ" >> $O/checks.txt
log sweep-done
awk -F, '$1=="sweep" && $7>0{b=1} END{exit b}' $O/ctrl/ctrl.csv || fail sweep-bits
# 2. sanitizer skipped (cannot attach under WSL2 WDDM; see provenance/v1.7/FINDINGS.md)
# 3. batch tests
$TB batch -o MUL_MAT -b CUDA0 > $O/batch-on.log 2>&1; echo "exit=$?" >> $O/batch-on.log
GGML_CUDA_BI_FAST_NB_KQ=0 $TB batch -o MUL_MAT -b CUDA0 > $O/batch-nbkq0.log 2>&1; echo "exit=$?" >> $O/batch-nbkq0.log
for f in batch-on batch-nbkq0; do echo "$f: $(grep -E 'tests passed' $O/$f.log | tail -1) $(grep exit= $O/$f.log)" >> $O/checks.txt; done
grep -q "exit=0" $O/batch-on.log || fail batch
log batch-done
# 4. draft model (all Q4_K/Q6_K): fixture plans, new kernel vs NB_KQ=0, and vs singles
for arm in on-new on-nbkq0; do
  mkdir -p $O/e1/$arm
  case $arm in
    on-new)   BI=1 $O/invariance $MD 99 8 cuda-on $O/e1/$arm > $O/e1-$arm.csv 2> $O/e1-$arm.log || fail e1-$arm ;;
    on-nbkq0) BI=1 GGML_CUDA_BI_FAST_NB_KQ=0 $O/invariance $MD 99 8 cuda-on $O/e1/$arm > $O/e1-$arm.csv 2> $O/e1-$arm.log || fail e1-$arm ;;
  esac
done
n=0; d=0; for f in $O/e1/on-nbkq0/*.f32; do n=$((n+1)); cmp -s $f $O/e1/on-new/$(basename $f) || d=$((d+1)); done
echo "draft e1 BI on: new vs nbkq0 files=$n differ=$d" >> $O/checks.txt
for a in on-new on-nbkq0; do echo "draft e1 $a: plans not bit-identical to own ref: $(awk -F, 'NR>1 && $8!=1' $O/e1-$a.csv | wc -l) of $(($(wc -l < $O/e1-$a.csv)-1))" >> $O/checks.txt; done
[ "$d" = "0" ] || fail e1-diff
bad=0
for s in singles whole c4 c8 pair4; do
  BI=1 $O/logit-dump-nctx $MD 99 $T $s $O/dumps/new-$s.bin 2> $O/dumps/new-$s.log || fail dump-new-$s
  BI=1 GGML_CUDA_BI_FAST_NB_KQ=0 $O/logit-dump-nctx $MD 99 $T $s $O/dumps/off-$s.bin 2> $O/dumps/off-$s.log || fail dump-off-$s
  if cmp -s $O/dumps/new-$s.bin $O/dumps/off-$s.bin; then r=identical; else r=DIFFERENT; bad=1; fi
  echo "draft plan $s new vs nbkq0 (BI on): $r" >> $O/checks.txt
  if cmp -s $O/dumps/new-singles.bin $O/dumps/new-$s.bin; then r=identical; else r=DIFFERENT; bad=1; fi
  echo "draft plan $s vs singles (BI on, new): $r" >> $O/checks.txt
  log "dumps $s"
done
sha256sum $O/dumps/*.bin > $O/dumps-SHA256SUMS; rm -f $O/dumps/*.bin
[ "$bad" = "0" ] || fail dumps
# 2B target sanity (Q8_0 only, gate code touched): 11-token fixture vs pr-bi-fafix files
mkdir -p $O/e1/t-on
BI=1 $O/invariance $MT 99 8 cuda-on $O/e1/t-on > $O/e1-t-on.csv 2> $O/e1-t-on.log || fail e1-t
n=0; d=0; for f in $Q/fa/e1/on/*.f32; do n=$((n+1)); cmp -s $f $O/e1/t-on/$(basename $f) || d=$((d+1)); done
echo "2B e1 BI on: vs pr-bi-fafix files=$n differ=$d" >> $O/checks.txt
[ "$d" = "0" ] || fail e1-t-diff
log checks-done
# 5. SASS op mix of the new kernels (supporting evidence; bit identity is the gate)
cuobjdump -sass $B/bin/libggml-cuda.so 2>/dev/null | awk '
  /Function :/ { f = ($3 ~ /bi_nb_q[46]_K_v3/) ? $3 : "" }
  f != "" { for (i = 1; i <= NF; i++) if ($i ~ /^(FMUL|FFMA|FADD|HMUL2|HFMA2|IDP|I2F)/) { sub(/;$/, "", $i); c[f" "$i]++ } }
  END { for (k in c) print c[k], k }' | sort -k2 > $O/sass-ops.txt
log sass-done
# 6. matched per-op perf, Q4_K only, 3 shuffled rounds
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > $O/gpu-start.txt
RE='type_a=q4_K,type_b=f32,m=[0-9]+,n=[28],k='
for rep in 1 2 3; do
  order=$(printf "default\nbi-new\nbi-mmq\n" | shuf --random-source=<(yes $rep) | tr '\n' ' ')
  echo "rep $rep order: $order" >> $O/perf-order.txt
  for arm in $order; do
    echo "$rep,$arm,$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader)" >> $O/perf-gpu.csv
    case $arm in
      default) $TB perf -o MUL_MAT -b CUDA0 -p "$RE" > $O/perf-$arm-r$rep.log 2>&1 ;;
      bi-new)  LLAMA_TEST_BATCH_INVARIANT_PERF=1 $TB perf -o MUL_MAT -b CUDA0 -p "$RE" > $O/perf-$arm-r$rep.log 2>&1 ;;
      bi-mmq)  LLAMA_TEST_BATCH_INVARIANT_PERF=1 GGML_CUDA_BI_FAST_NB_KQ=0 $TB perf -o MUL_MAT -b CUDA0 -p "$RE" > $O/perf-$arm-r$rep.log 2>&1 ;;
    esac
  done
done
log perf-done
# 7. BI speculative decoding, rs rollback: new kernel vs NB_KQ=0, 3 interleaved rounds
cd $O/spec
KS=0,1,3,7
for rep in 1 2 3; do
  for a in $(printf "new\noff\n" | shuf --random-source=<(yes $rep)); do
    echo "r$rep-bi-$a,$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader)" >> gpu.csv
    if [ $a = new ]; then BI=1 TRB=rs PREFIX=0 ../spec-bi $MT $MD r$rep-bi-$a $KS $T > r$rep-bi-$a.csv 2> r$rep-bi-$a.log || fail spec-$a
    else BI=1 GGML_CUDA_BI_FAST_NB_KQ=0 TRB=rs PREFIX=0 ../spec-bi $MT $MD r$rep-bi-$a $KS $T > r$rep-bi-$a.csv 2> r$rep-bi-$a.log || fail spec-$a; fi
    log "spec r$rep $a"
  done
done
python3 -I $S/v1.6/compare.py r1-bi-new.tok r1-bi-off.tok $SPEC_OLD/r1-bi-rs.tok > compare.txt
python3 -I $S/v1.6/spec_summary.py $O/spec > summary.txt
log done
touch $O/DONE
