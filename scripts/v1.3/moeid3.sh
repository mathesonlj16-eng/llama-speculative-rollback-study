#!/usr/bin/env bash
# Prediction: on STOCK sm_75, PR invariant mode is not batch-invariant for MUL_MAT_ID when MMF is refused:
#  - bf16 (MMF needs Ampere MMA) at any n
#  - f16 with m=2048 (>1024 rows) and n>128 tokens
# Controls: f16 m=768 GLU cases (MMF allowed up to 512 tokens), q8_0 (MMQ path).
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; X=$D/evidence-p61/f16-moe; mkdir -p $X
export SM75_HASH=1 SM75_SEED=1234 SM75_MOE_N=16,128,256; unset SM75_INV_NCOLS SM75_BATCH_TYPES
for t in f16 bf16 f32 q8_0; do f=$X/s75.$t.log
  SM75_MOE_TYPES=$t $D/build/pool-s75-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID > $f 2>&1; echo "exit=$?" >> $f
  echo "== $t: $(tail -1 $f)"
  sed 's/\x1b\[[0-9;]*m//g' $f | grep -oE '(\[[0-9]+/[0-9]+ differ, max diff [0-9.e+-]+\].*)?MUL_MAT_ID\(type_a=[a-z0-9_]+,n_mats=[0-9]+,n_used=[0-9]+,m=[0-9]+,k=[0-9]+,n=[0-9]+,glu=[01],shared=[01]\): *(OK|FAIL|not supported.*)' | sed -E 's/\[hb=[0-9a-f]+ hs=[0-9a-f]+\] *\[CONCAT\] ERR = [0-9.]+ > [0-9.]+ *//'
done
