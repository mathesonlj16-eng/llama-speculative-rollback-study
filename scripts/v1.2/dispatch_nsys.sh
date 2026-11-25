#!/usr/bin/env bash
# Runtime dispatch evidence: which kernels each arm launches, via nsys CUDA trace on a reduced perf matrix.
. $(dirname $0)/arms.sh
E=$D/evidence-p61; R=$E/dispatch; mkdir -p $R
export SM75_PERF_TYPES=q4_K,iq3_s SM75_PERF_M=4096 SM75_PERF_K=4096 SM75_PERF_N=1,8,16,64,512
env | grep ^SM75_ > $R/env.txt
for a in s75m0 s75m2 s75pron fm0 fm2 fpron nm0; do
  while busy; do sleep 15; done
  t0=$(date +%s.%N)
  nsys profile -t cuda -o $R/$a --force-overwrite true bash -c ". $(dirname $0)/arms.sh; exe $a perf" > $R/$a.log 2>&1; rc=$?
  echo "exit=$rc wall_s=$(awk "BEGIN{print $(date +%s.%N)-$t0}")" >> $R/$a.log
  nsys stats -q -r cuda_gpu_kern_sum -f csv -o $R/$a $R/$a.nsys-rep > /dev/null 2>&1
done
echo DONE > $R/DONE
