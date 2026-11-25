#!/usr/bin/env bash
# Guarded perf sweep, same acceptance rule as run_perf_curve2.sh (every non-P8 clock sample at 6120 MHz mem),
# plus a CONTINUOUS 1 s compiler-activity sampler during each run (any busy sample rejects the run).
# One labeled warm run per arm, then ROUNDS measured rounds with the arm order rotated by one each round.
# usage: run_perf_p61.sh TAG ROUNDS "ARMS"
. $(dirname $0)/arms.sh
E=$D/evidence-p61; R=$E/perf/$1; mkdir -p $R; L=$R/guard.log
export SM75_PERF_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_PERF_M=4096 SM75_PERF_K=4096 SM75_PERF_N=1,2,4,8,16,24,32,48,64,128,256,512
env | grep -E '^SM75_|^CUDA_CACHE' > $R/env.txt
one() { local arm=$1 tag=$2 try
  for try in 1 2 3 4; do
    while busy; do sleep 15; done
    local f=$R/$arm.$tag.try$try.log s=$R/$arm.$tag.try$try.clk c=$R/$arm.$tag.try$try.busy
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > $s 2>/dev/null & local sp=$!
    ( while :; do if busy; then echo "$(date -Is) BUSY $(pgrep -fa 'c[c]1plus|nv[c]c |[c]icc|[p]txas' | head -2 | tr '\n' ' ')"; else echo "$(date -Is) idle"; fi; sleep 1; done ) > $c 2>&1 & local bp=$!
    echo "arm=$arm tag=$tag try=$try $(date -Is)" > $f; exe $arm perf >> $f 2>&1; echo "exit=$?" >> $f
    kill $sp $bp; wait $sp $bp 2>/dev/null
    local bad=$(awk -F', ' '$2!="P8" && $4!="6120 MHz"' $s | wc -l) nb=$(grep -c BUSY $c)
    if [ "$bad" = 0 ] && [ "$nb" = 0 ] && grep -q '^exit=0' $f; then
      echo "$arm.$tag try$try CLEAN clk_samples=$(wc -l < $s) busy_samples=$(wc -l < $c)" >> $L; cp $f $R/$arm.$tag.log; return; fi
    echo "$arm.$tag try$try REJECT bad_clk=$bad busy=$nb exit=$(tail -1 $f)" >> $L; sleep 30
  done
  echo "$arm.$tag GAVE UP" >> $L; }
A=($3); n=${#A[@]}
for a in "${A[@]}"; do one $a warm; done
for r in $(seq 1 $2); do o=""; for i in $(seq 0 $((n-1))); do o="$o ${A[$(( (i + r - 1) % n ))]}"; done
  echo "round r$r order:$o" >> $L; for a in $o; do one $a r$r; done; done
echo DONE >> $L
