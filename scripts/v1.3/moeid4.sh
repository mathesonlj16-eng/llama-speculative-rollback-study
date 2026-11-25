#!/usr/bin/env bash
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; W=$D/worktrees/pr-pool; X=$D/evidence-p61/f16-moe
export CCACHE_DIR=${CCACHE_DIR:-$HOME/.cache/ccache}
python3 -I "$(dirname "$0")/edit_f32sort.py" $W/ggml/src/ggml-cuda/ggml-cuda.cu || exit 1
git -C $W diff -- ggml/src/ggml-cuda/ggml-cuda.cu > $X/f32sort.diff
t0=$(date +%s); cmake --build $D/build/pool-s75-pr --target test-backend-ops -j8 > $X/build-f32sort.log 2>&1 || { echo build-fail; tail -20 $X/build-f32sort.log; exit 1; }
echo "rebuilt in $(( $(date +%s)-t0 )) s"
export SM75_HASH=1 SM75_SEED=1234 SM75_MOE_N=16,128,256; unset SM75_INV_NCOLS SM75_BATCH_TYPES
for mode in off on; do for t in f16 bf16; do f=$X/s75.$t.f32sort-$mode.log
  if [ $mode = on ]; then export SM75_MOE_F32SORT=1; else unset SM75_MOE_F32SORT; fi
  SM75_MOE_TYPES=$t $D/build/pool-s75-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID > $f 2>&1; echo "exit=$?" >> $f
  c=$(sed 's/\x1b\[[0-9;]*m//g' $f); echo "$t f32sort=$mode: ok=$(grep -cE 'MUL_MAT_ID\(.*\): *OK' <<< "$c") fail=$(grep -cE 'MUL_MAT_ID\(.*\): *FAIL' <<< "$c") $(tail -1 $f)"
done; done
# does default (unset) still match the earlier pre-patch run bit for bit?
python3 -I - <<'EOF'
import re, os
X = os.path.join(os.environ.get("SM75_ROOT", os.path.expanduser("~/research/sm75-batch-invariance")), "evidence-p61/f16-moe")
pat = re.compile(r"\[hb=([0-9a-f]+) hs=([0-9a-f]+)\].*?(MUL_MAT_ID\([^)]*\))")
ansi = re.compile(r"\x1b\[[0-9;]*m")
def load(f): return {m.group(3): m.group(1, 2) for m in (pat.search(ansi.sub("", l)) for l in open(f, errors="replace")) if m}
for t in ("f16", "bf16"):
    a, b = load(f"{X}/s75.{t}.log"), load(f"{X}/s75.{t}.f32sort-off.log")
    print(t, "default after patch == before patch:", sum(a[n] == b[n] for n in a), "/", len(a))
EOF
