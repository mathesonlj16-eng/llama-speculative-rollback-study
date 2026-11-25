#!/usr/bin/env bash
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; X=$D/evidence-p61/jfree; L=$X/log.txt; st() { echo "$(date -Is) $*" | tee -a $L; }
for b in pool-d4f-pr pool-s75-pr; do for nc in unset 0 64; do
  if [ $nc = unset ]; then unset SM75_INV_NCOLS; else export SM75_INV_NCOLS=$nc; fi
  SM75_INV_LOG=1 SM75_SEED=1234 SM75_BATCH_TYPES=q4_K,iq3_s SM75_BATCH_M=256 SM75_BATCH_K=1024 SM75_BATCH_N=1,9,16,64,512 \
    $D/build/$b/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT > $X/jbest.$b.$nc.raw 2>&1
  st "J choice $b ncols=$nc: $(grep -o 'ne11=[0-9]* inv=1 J_best=[0-9]*' $X/jbest.$b.$nc.raw | sort -u | sort -t= -k2 -n | sed 's/ inv=1 J_best=/->J/' | tr '\n' ' ')"
done; done
unset SM75_INV_NCOLS
P=$X/perf; B=$D/build/pool-d4f-pr/bin/test-backend-ops
export SM75_PERF_TYPES=iq3_s,iq4_xs,q4_K,q5_K,q8_0 SM75_PERF_M=4096 SM75_PERF_K=4096 SM75_PERF_N=1,16,128,512
arm() { case $1 in
  def)  env -u SM75_PERF_INVARIANT -u SM75_INV_NCOLS $B perf -b CUDA0 -o MUL_MAT ;;
  pin)  env -u SM75_INV_NCOLS SM75_PERF_INVARIANT=1 $B perf -b CUDA0 -o MUL_MAT ;;
  free) env SM75_PERF_INVARIANT=1 SM75_INV_NCOLS=0 $B perf -b CUDA0 -o MUL_MAT ;;
  esac; }
arm pin > $P/pin.warm.log 2>&1
A=(pin def free)
for r in 1 2 3; do o=""; for i in 0 1 2; do o="$o ${A[$(( (i + r - 1) % 3 ))]}"; done
  for a in $o; do f=$P/$a.s$r
    nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > $f.clk 2>/dev/null & sp=$!
    arm $a > $f.log 2>&1; echo "exit=$?" >> $f.log; kill $sp; wait $sp 2>/dev/null
    st "perf2 $a s$r: $(tail -1 $f.log) memclk_bad=$(awk -F', ' '$2!="P8" && $4!="6120 MHz"' $f.clk | wc -l) sm_min=$(awk -F', ' '{print $3+0}' $f.clk | sort -n | head -1) tmax=$(awk -F', ' '{print $5+0}' $f.clk | sort -n | tail -1)"
  done; done
st DONE2
