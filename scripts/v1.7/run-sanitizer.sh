#!/bin/bash
# Archived driver, as run on 2026-11-26 (see REPRODUCE.md). Only the paths were changed for publication:
# SM75_ROOT = research root (worktrees/, build/, models/, qwen-bi/); S = this repository's scripts/ folder.
export PATH=/usr/local/cuda-13.2/bin:$PATH
R=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}/scratch-mmq-n1
L=$R/nbkq3/ctrl/lib; Q8=$R/nb/ctrl/ctrl
O=$R/nbkq3/san; mkdir -p $O; cd $O; : > summary.txt
run() { # name tool cmd...
  local n=$1 t=$2; shift 2
  local s=$(date +%s)
  timeout 1200 compute-sanitizer --tool $t "$@" > $n-$t.log 2>&1; local e=$?
  echo "$n $t exit=$e $(($(date +%s)-s))s :: $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' $n-$t.log | tail -1 | sed 's/=========//')" >> summary.txt
}
for t in memcheck racecheck initcheck synccheck; do
  run q4K-1024x100 $t $L q4_K 1024 100 1 0 san
  run q4K-3584x100 $t $L q4_K 3584 100 1 0 san
  run q6K-1024x100 $t $L q6_K 1024 100 1 0 san
  run q6K-3584x100 $t $L q6_K 3584 100 1 0 san
  run q8-2048x100  $t $Q8 2048 100 1 0 san
done
cat summary.txt
