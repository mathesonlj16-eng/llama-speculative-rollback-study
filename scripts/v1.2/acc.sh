#!/usr/bin/env bash
# Accuracy vs CPU (standard test-backend-ops eval) for the dp4a-switched MMQ, since perf mode does not check results.
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; R=$E/accuracy; mkdir -p $R
P='type_a=(q4_K|iq3_s|iq4_xs|q5_K|q8_0),type_b=f32'
for b in d4f-fw:fm0 d4f-pr:pr fixed-warp:s75m0; do bd=${b%%:*}; tag=${b#*:}
  t0=$(date +%s)
  GGML_CUDA_MMVQ_INV=0 $D/build/$bd/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p "$P" > $R/$tag.log 2>&1; rc=$?
  echo "$tag build=$bd exit=$rc ok=$(grep -c 'OK' $R/$tag.log) fail=$(grep -c 'FAIL' $R/$tag.log) secs=$(( $(date +%s)-t0 ))" | tee -a $R/summary.txt
  grep 'FAIL' $R/$tag.log | head -3 | cut -c1-200
done
