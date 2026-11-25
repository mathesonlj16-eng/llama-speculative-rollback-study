#!/usr/bin/env bash
# Tile-width test: does PR #32081's invariant mode stay bitwise batch-invariant if the MMQ tile width J
# is not pinned (SM75_INV_NCOLS=0: J follows N) or pinned wide (SM75_INV_NCOLS=64)? Unset = PR behaviour (1).
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; W=$D/worktrees/pr-pool; X=$E/jfree
mkdir -p $X; L=$X/log.txt; st() { echo "$(date -Is) $*" | tee -a $L; }
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
F=$W/ggml/src/ggml-cuda/mmq.cu
python3 -I "$(dirname "$0")/edit_mmq.py" $F || exit 1
n=$(grep -c 'SM75_INV_NCOLS\|SM75_INV_LOG' $F); st "mmq.cu edits present: $n (expect 3)"; [ "$n" -ge 3 ] || exit 1
git -C $W diff -- ggml/src/ggml-cuda/mmq.cu > $X/mmq-jfree.diff
git -C $W diff > $X/worktree-pr-jfree.full.diff
for b in pool-d4f-pr pool-s75-pr; do t0=$(date +%s)
  cmake --build $D/build/$b --target test-backend-ops -j8 > $X/build-$b.log 2>&1 || { st "build FAIL $b"; exit 1; }
  st "rebuilt $b in $(( $(date +%s)-t0 )) s; compiled units: $(grep -c 'Building' $X/build-$b.log)"; done
# which J does each setting pick? (one quick run, log only)
for b in pool-d4f-pr; do for nc in unset 0 64; do
  if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  SM75_INV_LOG=1 SM75_SEED=1234 SM75_BATCH_TYPES=q4_K SM75_BATCH_M=256 SM75_BATCH_K=1024 SM75_BATCH_N=9,16,64,512 \
    $D/build/$b/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT 2>&1 | grep -o 'ne11=[0-9]* inv=1 J_best=[0-9]*' | sort -u -t= -k2,2n | tr '\n' ' ' > $X/jbest.$nc.txt
  st "J choice ncols=$nc: $(cat $X/jbest.$nc.txt)"; done; done
unset SM75_INV_NCOLS
export SM75_BATCH_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_BATCH_M=256,1024 SM75_BATCH_K=1024,4096,5120 SM75_HASH=1
SMALL=1,2,3,4,5,6,7,8,9,10; BIG=9,10,11,12,13,14,15,16,24,32,48,64,128,256,512
run() { local b=$1 nc=$2 grid=$3 seed=$4 f=$X/$1.nc$2.$3.$4.log
  if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  if [ $seed = rand ]; then unset SM75_SEED; else export SM75_SEED=$seed; fi
  if [ $grid = small ]; then export SM75_BATCH_N=$SMALL; else export SM75_BATCH_N=$BIG; fi
  echo "build=$b ncols=$nc grid=$grid seed=$seed $(date -Is)" > $f
  $D/build/$b/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT >> $f 2>&1; echo "exit=$?" >> $f
  st "$b ncols=$nc $grid $seed: $(tail -1 $f) ok=$(grep -c 'MUL_MAT.*OK' $f) fail=$(grep -c 'MUL_MAT.*FAIL' $f)"; }
for b in pool-d4f-pr pool-s75-pr; do
  for nc in 0 unset 64; do for grid in big small; do run $b $nc $grid 1234; done; done
  run $b 0 big rand; run $b 0 small rand
done
unset SM75_INV_NCOLS SM75_SEED SM75_HASH
st "CORRECTNESS DONE"
# short dp4a timing: default MMQ, PR invariant (J pinned), invariant with free J. 1 unguarded warm, 3 rotated rounds, clocks logged.
P=$X/perf; mkdir -p $P
export SM75_PERF_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_PERF_M=4096 SM75_PERF_K=4096 SM75_PERF_N=1,16,128,512
B=$D/build/pool-d4f-pr/bin/test-backend-ops
arm() { case $1 in
  def)   env -u SM75_PERF_INVARIANT -u SM75_INV_NCOLS $B perf -b CUDA0 -o MUL_MAT ;;
  pin)   env SM75_PERF_INVARIANT=1 -u SM75_INV_NCOLS $B perf -b CUDA0 -o MUL_MAT ;;
  free)  env SM75_PERF_INVARIANT=1 SM75_INV_NCOLS=0 $B perf -b CUDA0 -o MUL_MAT ;;
  esac; }
for a in def pin free; do arm $a > $P/$a.warm.log 2>&1; done
A=(def pin free)
for r in 1 2 3; do o=""; for i in 0 1 2; do o="$o ${A[$(( (i + r - 1) % 3 ))]}"; done
  for a in $o; do
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > $P/$a.r$r.clk 2>/dev/null & sp=$!
    arm $a > $P/$a.r$r.log 2>&1; echo "exit=$?" >> $P/$a.r$r.log; kill $sp; wait $sp 2>/dev/null
    st "perf $a r$r: $(tail -1 $P/$a.r$r.log) memclk_bad=$(awk -F', ' '$2!="P8" && $4!="6120 MHz"' $P/$a.r$r.clk | wc -l) sm_min=$(awk -F', ' '{print $3+0}' $P/$a.r$r.clk | sort -n | head -1)"
  done; done
st DONE
