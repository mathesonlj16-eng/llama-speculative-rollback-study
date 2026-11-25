#!/usr/bin/env bash
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; W=$D/worktrees/pr-pool; X=$D/evidence-p61/jfree-moe2; mkdir -p $X; L=$X/log.txt
st() { echo "$(date -Is) $*" | tee -a $L; }
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
T=$W/tests/test-backend-ops.cpp
python3 -I "$(dirname "$0")/edit_moe.py" $T && python3 -I "$(dirname "$0")/edit_moe2.py" $T || { st "edit failed"; exit 1; }
st "moe edits: $(grep -c 'SM75_MOE_TYPES\|sm75_seed_ids' $T)"
git -C $W diff -- tests/test-backend-ops.cpp > $X/test-backend-ops.full.diff
for b in pool-d4f-pr pool-s75-pr; do t0=$(date +%s)
  cmake --build $D/build/$b --target test-backend-ops -j8 > $X/build-$b.log 2>&1 || { st "build FAIL $b"; tail -20 $X/build-$b.log >> $L; exit 1; }
  st "rebuilt $b in $(( $(date +%s)-t0 )) s"; done
export SM75_HASH=1 SM75_SEED=1234 SM75_MOE_TYPES=q4_K,q8_0,q6_K,iq4_xs SM75_MOE_N=16,64,128,256
unset SM75_BATCH_TYPES
# which J does MoE MMQ pick?
for nc in 0 unset; do if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  SM75_MOE_TYPES=q4_K SM75_INV_LOG=1 $D/build/pool-d4f-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID 2>&1 | grep -o 'ne11=[0-9]* inv=1 J_best=[0-9]*' | sort -u > $X/jbest.$nc.txt
  st "J choice ncols=$nc: $(awk '{print $1"->"$3}' $X/jbest.$nc.txt | sort -t'=' -k2 -n | tr '\n' ' ')"; done
run() { local b=$1 nc=$2 rep=$3 f=$X/$1.nc$2.$3.log
  if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  echo "build=$b ncols=$nc rep=$rep $(date -Is)" > $f
  $D/build/$b/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID >> $f 2>&1; echo "exit=$?" >> $f
  st "$b ncols=$nc $rep: $(tail -1 $f) ok=$(grep -c 'OK' $f) fail=$(grep -c 'FAIL' $f)"; }
for b in pool-d4f-pr pool-s75-pr; do run $b unset a; run $b unset b; run $b 0 a; done
st DONE
