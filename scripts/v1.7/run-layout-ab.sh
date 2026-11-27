#!/bin/bash
# Archived driver, as run on 2026-11-26 (see REPRODUCE.md). Only the paths were changed for publication:
# SM75_ROOT = research root (worktrees/, build/, models/, qwen-bi/); S = this repository's scripts/ folder.
# Matched A/B of Q4_K N>=2 staging layouts in one build (GGML_CUDA_BI_NB_Q4K_LAYOUT=f4|compact|unset=hybrid):
# quick bit gate per layout, then BI speculative decoding with the three layouts interleaved, 3 rounds.
set -u
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; Q=$R/qwen-bi
S=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
W=$R/worktrees/mmq-n1-fa; B=$R/build/mmq-n1-fa
MT=$R/models/Qwen3.5-2B-Q8_0.gguf; MD=$R/models/Qwen3.5-0.8B-Q4_K_M.gguf; T=$Q/src/ngram-token-fixtures.tsv
SPEC_OLD=$R/scratch-mmq-n1/spec/full
O=$R/scratch-mmq-n1/nbkq-ab
mkdir -p $O/spec
log() { echo "$(date -Is) $*" >> $O/timeline.txt; }
fail() { log "FAIL $*"; touch $O/FAIL; exit 1; }
rm -f $O/DONE $O/FAIL $O/checks.txt
export PATH=/usr/local/cuda-13.2/bin:$PATH
log "build start"
cmake --build $B --target llama -j "$(nproc)" > $O/build.log 2>&1 || fail build
g++ -O2 -std=c++17 -I$W/include -I$W/ggml/include $S/v1.6/spec-bi.cpp -o $O/spec-bi -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin > $O/compile.log 2>&1 || fail compile-spec
nvcc -O2 -arch=sm_75 -std=c++17 -I$W/ggml/include $S/v1.7/bi_nb_kq_lib.cu -o $O/lib \
    -L$B/bin -lggml -lggml-base -lggml-cuda -Xlinker -rpath=$B/bin >> $O/compile.log 2>&1 || fail compile-lib
log compiled
# bit gate per layout: every column of N=1..8 vs N=16 forced MMQ, two draft shapes (one with a tail chunk)
for lay in f4 compact hybrid; do
  for s in 1024:2048 3584:1024; do
    K=${s%%:*}; MM=${s##*:}
    if [ $lay = hybrid ]; then env -u GGML_CUDA_BI_NB_Q4K_LAYOUT $O/lib q4_K $K $MM 2 0245 $lay >> $O/gate.csv 2>> $O/gate.summary || fail gate-$lay
    else GGML_CUDA_BI_NB_Q4K_LAYOUT=$lay $O/lib q4_K $K $MM 2 0245 $lay >> $O/gate.csv 2>> $O/gate.summary || fail gate-$lay; fi
  done
done
echo "gate: $(awk -F, '$1!="label"{s+=$7; c+=$8} END{print "bad "s+0" of "c+0}' $O/gate.csv)" >> $O/checks.txt
awk -F, '$1!="label" && $7>0{b=1} END{exit b}' $O/gate.csv || fail gate-bits
log gate-done
# speculative A/B, rs rollback, 3 interleaved rounds
cd $O/spec
KS=0,1,3,7
for rep in 1 2 3; do
  for a in $(printf "hybrid\nf4\ncompact\n" | shuf --random-source=<(yes $rep)); do
    echo "r$rep-bi-$a,$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader),$(free -m | awk 'NR==2{print $3"MB used, "$7"MB avail"}')" >> gpu.csv
    if [ $a = hybrid ]; then env -u GGML_CUDA_BI_NB_Q4K_LAYOUT BI=1 TRB=rs PREFIX=0 ../spec-bi $MT $MD r$rep-bi-$a $KS $T > r$rep-bi-$a.csv 2> r$rep-bi-$a.log || fail spec-$a
    else GGML_CUDA_BI_NB_Q4K_LAYOUT=$a BI=1 TRB=rs PREFIX=0 ../spec-bi $MT $MD r$rep-bi-$a $KS $T > r$rep-bi-$a.csv 2> r$rep-bi-$a.log || fail spec-$a; fi
    log "spec r$rep $a"
  done
done
python3 -I $S/v1.6/compare.py r1-bi-hybrid.tok r1-bi-f4.tok r1-bi-compact.tok $SPEC_OLD/r1-bi-rs.tok > compare.txt
python3 -I $S/v1.6/spec_summary.py $O/spec > summary.txt
log done
touch $O/DONE
