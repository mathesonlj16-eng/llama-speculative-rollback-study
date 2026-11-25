#!/usr/bin/env bash
D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; X=$D/evidence-p61/jfree-cov; T=$D/worktrees/pr-pool/tests/test-backend-ops.cpp
export SM75_HASH=1 SM75_SEED=1234; unset SM75_INV_NCOLS SM75_BATCH_TYPES SM75_BATCH_M SM75_BATCH_K SM75_BATCH_N
for r in a b; do $D/build/pool-s75-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID > $X/repeat.s75.pin.$r.log 2>&1; echo "exit=$?" >> $X/repeat.s75.pin.$r.log; done
for r in a b; do $D/build/pool-d4f-pr/bin/test-backend-ops batch -b CUDA0 -o MUL_MAT_ID > $X/repeat.d4f.pin.$r.log 2>&1; echo "exit=$?" >> $X/repeat.d4f.pin.$r.log; done
python3 -I - <<'EOF'
import re, os
X = os.path.expanduser("~/research/sm75-batch-invariance/evidence-p61/jfree-cov")
ansi = re.compile(r"\x1b\[[0-9;]*m"); pat = re.compile(r"\[hb=([0-9a-f]+) hs=([0-9a-f]+)\].*?(MUL_MAT_ID\([^)]*\)):\s+(\S+)")
def load(f): return {m.group(3): (m.group(1), m.group(4)) for m in (pat.search(ansi.sub("", l)) for l in open(f, errors="replace")) if m}
for b in ("s75", "d4f"):
    a, c = load(f"{X}/repeat.{b}.pin.a.log"), load(f"{X}/repeat.{b}.pin.b.log")
    com = [n for n in a if n in c]
    print(b, "pinned run A vs B: cases", len(com), "batch-hash equal", sum(a[n][0] == c[n][0] for n in com),
          "| fails A", sum(v[1].startswith("FAIL") for v in a.values()), "B", sum(v[1].startswith("FAIL") for v in c.values()))
    d = [n for n in com if a[n][0] != c[n][0]]; print("   differing types:", sorted({n.split(",")[0] for n in d}))
EOF
n=$(grep -n "struct test_mul_mat_id_batch_inv" $T | cut -d: -f1); sed -n "$n,$((n+80))p" $T | grep -nE "rand|shuffle|mt19937|random|init_tensor|ids" | head -12
