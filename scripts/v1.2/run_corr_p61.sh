#!/usr/bin/env bash
# Correctness (batch invariance) grids on the Pascal-code builds. usage: run_corr_p61.sh TAG NLIST "ARMS"
. $(dirname $0)/arms.sh
E=$D/evidence-p61; R=$E/runs/$1; mkdir -p $R
export SM75_BATCH_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_BATCH_M=256,1024 SM75_BATCH_K=1024,4096,5120 SM75_BATCH_N=$2 SM75_HASH=1
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > $R/gpu-before.csv
env | grep ^SM75_ > $R/env.txt
one() { local arm=$1 seed=$2; local f=$R/$arm.$seed.r1.log
  if [ "$seed" = rand ]; then unset SM75_SEED; else export SM75_SEED=$seed; fi
  echo "arm=$arm seed=$seed rep=1 $(date -Is)" > $f; exe $arm batch >> $f 2>&1; echo "exit=$?" >> $f; }
for seed in 1234 rand; do for a in $3; do one $a $seed; done; done
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate --format=csv > $R/gpu-after.csv
echo RUNDONE > $R/DONE
