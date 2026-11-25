#!/usr/bin/env bash
# Pipeline: builds -> static check -> nsys dispatch -> correctness (gate) -> guarded perf. Strictly sequential (no compile during timing).
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61; S=$E/scripts; P=$E/pipeline.log
st() { echo "$(date -Is) $*" >> $P; }
{ nvidia-smi; /usr/local/cuda-13.2/bin/nvcc --version; nsys --version; uname -a; } > $E/environment.txt 2>&1
st START
bash $S/build_p61.sh || { st "BUILD FAILED"; echo FAILED > $E/STATUS; exit 1; }; st builds-done
bash $S/static_check.sh; st static-done
bash $S/dispatch_nsys.sh; st dispatch-done
bash $S/run_corr_p61.sh small 1,2,3,4,5,6,7,8,9,10 "fm0 fm1 fm2 fpron"; st corr-small-done
bash $S/run_corr_p61.sh bigN 9,10,11,12,13,14,15,16,24,32,48,64,128,256,512 "fm0 fm1 fm2 fpron"; st corr-bigN-done
fail=0
# Gate: chunked arm on both grids; PR arm only on the small grid (the only range where its sm_75 build was tested).
for f in $E/runs/*/fm2.*.log $E/runs/small/fpron.*.log; do
  grep -q '^exit=0' $f && ! grep -q 'FAIL' $f || { st "GATE FAIL $f"; fail=1; }; done
for f in $E/runs/bigN/fpron.*.log; do grep -q '^exit=0' $f && ! grep -q 'FAIL' $f || st "NOTE (not gated) PR arm fails at large N: $f"; done
[ $fail = 1 ] && { st "STOPPED before perf: invariant arm correctness failure"; echo GATE_FAILED > $E/STATUS; exit 1; }
st gate-pass
bash $S/run_perf_p61.sh main 10 "s75m0 s75m2 s75pron fm0 fm2 fpron nm0"; st perf-done
echo DONE > $E/STATUS; st END
