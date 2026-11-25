#!/usr/bin/env bash
# Coverage for the tile-width result: (1) the PR's own full batch suite, (2) fallback M, odd K, more quant types.
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; X=$D/evidence-p61/jfree-cov; mkdir -p $X; L=$X/log.txt
st() { echo "$(date -Is) $*" | tee -a $L; }
grep -n "sm75_types" $D/worktrees/pr-pool/tests/test-backend-ops.cpp | head -2 >> $L
export SM75_HASH=1 SM75_SEED=1234
summ() { echo "$(tail -1 $1) ok=$(grep -c '\bOK\b' $1) fail=$(grep -c 'FAIL' $1) unsupported=$(grep -ci 'not supported' $1)"; }
run() { local tag=$1 b=$2 nc=$3; shift 3; local f=$X/$tag.$b.nc$nc.log
  if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  echo "tag=$tag build=$b ncols=$nc $(date -Is)" > $f
  $D/build/$b/bin/test-backend-ops batch -b CUDA0 "$@" >> $f 2>&1; echo "exit=$?" >> $f
  st "$tag $b ncols=$nc: $(summ $f)"; }
# (1) PR's default batch suite (all ops), SM75_BATCH_* unset
unset SM75_BATCH_TYPES SM75_BATCH_M SM75_BATCH_K SM75_BATCH_N
for b in pool-d4f-pr pool-s75-pr; do for nc in 0 unset; do run suite $b $nc; done; done
# (2a) more types + fallback M (not a multiple of 128), N up to 512
export SM75_BATCH_TYPES=q2_K,q3_K,q6_K,q4_0,q4_1,q5_0,iq2_xxs,iq2_xs,iq3_xxs,iq4_nl,iq3_s,q4_K,q8_0 SM75_BATCH_M=256,1000 SM75_BATCH_K=1024,4352 SM75_BATCH_N=9,16,24,64,128,512
for b in pool-d4f-pr pool-s75-pr; do for nc in 0 unset; do run types $b $nc -o MUL_MAT; done; done
# (2b) K not a multiple of 256 (32-element block types only)
export SM75_BATCH_TYPES=q4_0,q8_0,q5_0,iq4_nl SM75_BATCH_M=1000,4096 SM75_BATCH_K=1056,2080 SM75_BATCH_N=1,9,16,64,512
for b in pool-d4f-pr pool-s75-pr; do for nc in 0 unset; do run oddk $b $nc -o MUL_MAT; done; done
st DONE
