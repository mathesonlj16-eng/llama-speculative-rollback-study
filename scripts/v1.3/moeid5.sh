#!/usr/bin/env bash
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; W=$D/worktrees/pr-pool; X=$D/evidence-p61/f16-moe
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
python3 -I "$(dirname "$0")/edit_nommf.py" $W/ggml/src/ggml-cuda/ggml-cuda.cu || exit 1
git -C $W diff -- ggml/src/ggml-cuda/ggml-cuda.cu > $X/moe-fix.diff
t0=$(date +%s); cmake --build $D/build/pool-s75-pr --target test-backend-ops -j8 > $X/build-nommf.log 2>&1 || { echo build-fail; tail -20 $X/build-nommf.log; exit 1; }
echo "rebuilt in $(( $(date +%s)-t0 )) s"
export SM75_HASH=1 SM75_SEED=1234 SM75_MOE_N=16,128,256; unset SM75_INV_NCOLS SM75_BATCH_TYPES
for t in f16 bf16 f32 q8_0; do f=$X/s75.$t.fix-both.log
  SM75_MOE_F32SORT=1 SM75_MOE_NOMMF=1 SM75_MOE_TYPES=$t $D/build/pool-s75-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID > $f 2>&1; echo "exit=$?" >> $f
  c=$(sed 's/\x1b\[[0-9;]*m//g' $f); echo "$t both fixes: ok=$(grep -cE 'MUL_MAT_ID\(.*\): *OK' <<< "$c") fail=$(grep -cE 'MUL_MAT_ID\(.*\): *FAIL' <<< "$c") $(tail -1 $f)"
done
# PR's own default suite with both fixes, to check nothing else regresses
unset SM75_MOE_TYPES SM75_MOE_N; f=$X/s75.suite.fix-both.log
SM75_MOE_F32SORT=1 SM75_MOE_NOMMF=1 $D/build/pool-s75-pr/bin/test-backend-ops batch -b CUDA0 > $f 2>&1; echo "exit=$?" >> $f
c=$(sed 's/\x1b\[[0-9;]*m//g' $f); echo "PR suite both fixes: ok=$(grep -cE '\): *OK' <<< "$c") fail=$(grep -cE '\): *FAIL' <<< "$c") $(tail -1 $f)"
# rough cost: F16 MoE perf, default vs both fixes, invariant on (one run each, informational only)
