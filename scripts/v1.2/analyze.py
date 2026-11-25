#!/usr/bin/env python3
"""Parse the guarded dp4a-switch sweep: per-run timings + per-run clock stats, medians, ratios, clock-strict sensitivity."""
import re, sys, os, glob, csv, statistics as st, collections
R = sys.argv[1]; OUT = sys.argv[2]
guard = open(f"{R}/guard.log").read().splitlines()
clean = {}  # (arm, tag) -> try
for l in guard:
    m = re.match(r"(\w+)\.(r\d+) try(\d) CLEAN", l)
    if m: clean[(m[1], m[2])] = m[3]
rx = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=4096,n=(\d+),k=4096.*?([\d.]+) us/run")
rows, runs = [], []
for (arm, tag), t in sorted(clean.items()):
    txt = open(f"{R}/{arm}.{tag}.log").read()
    clk = [l.split(", ") for l in open(f"{R}/{arm}.{tag}.try{t}.clk") if l.strip()]
    act = [c for c in clk if c[1] != "P8"]
    sm = [int(c[2].split()[0]) for c in act]
    flags = collections.Counter(int(c[6], 16) for c in act)
    thermal = sum(v for k, v in flags.items() if k & 0x60)      # SW/HW thermal slowdown
    pcap = sum(v for k, v in flags.items() if k & 0x04)         # SW power cap
    temps = [int(c[4]) for c in act]
    strict = bool(sm) and min(sm) >= 1785 and thermal == 0
    runs.append(dict(arm=arm, tag=tag, try_=t, n_samples=len(act), sm_min=min(sm), sm_median=st.median(sm),
                     temp_max=max(temps), thermal_samples=thermal, powercap_samples=pcap, strict=strict))
    for ty, n, us in rx.findall(txt):
        rows.append(dict(arm=arm, tag=tag, type=ty, N=int(n), us=float(us), strict=strict))
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/rounds.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/run-clocks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)

def med(subset):
    d = collections.defaultdict(list)
    for r in subset: d[(r["arm"], r["type"], r["N"])].append(r["us"])
    return {k: (st.median(v), min(v), max(v), len(v)) for k, v in d.items()}
M, MS = med(rows), med([r for r in rows if r["strict"]])
arms = ["s75m0", "s75m2", "s75pron", "fm0", "fm2", "fpron", "nm0"]
types = sorted({r["type"] for r in rows}); Ns = sorted({r["N"] for r in rows})
with open(f"{OUT}/medians.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["type", "N"] + [f"{a}_us" for a in arms] + [f"{a}_n" for a in arms] + [f"{a}_spread_pct" for a in arms])
    for ty in types:
        for n in Ns:
            g = [M.get((a, ty, n)) for a in arms]
            w.writerow([ty, n] + [round(x[0], 2) if x else "" for x in g] + [x[3] if x else 0 for x in g]
                       + [round(100 * (x[2] - x[1]) / x[0], 2) if x else "" for x in g])
ratios = [("fpron/fm2", "dp4a PR-MMQ vs chunked (switched)"), ("s75pron/s75m2", "MMA PR-MMQ vs chunked (sm_75)"),
          ("fpron/s75pron", "switch effect on PR-MMQ"), ("fm0/s75m0", "switch effect on default"),
          ("fm2/s75m2", "CONTROL: chunked, should be ~1"), ("fm2/fm0", "chunked vs default (switched)"),
          ("s75m2/s75m0", "chunked vs default (sm_75)"), ("nm0/fm0", "no-FORCE (cuBLAS>=64) vs FORCE default")]
def ratio_table(Mx):
    out = {}
    for rk, _ in ratios:
        a, b = rk.split("/")
        out[rk] = {(ty, n): Mx[(a, ty, n)][0] / Mx[(b, ty, n)][0] for ty in types for n in Ns if (a, ty, n) in Mx and (b, ty, n) in Mx}
    return out
RT, RTS = ratio_table(M), ratio_table(MS)
with open(f"{OUT}/ratios.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["type", "N"] + [k for k, _ in ratios] + [k + "_strict" for k, _ in ratios])
    for ty in types:
        for n in Ns:
            w.writerow([ty, n] + [round(RT[k].get((ty, n), float("nan")), 4) for k, _ in ratios] + [round(RTS[k].get((ty, n), float("nan")), 4) for k, _ in ratios])
print(f"clean measured runs: {len(runs)}; strict-clock runs: {sum(r['strict'] for r in runs)}; "
      f"sm_min range {min(r['sm_min'] for r in runs)}-{max(r['sm_min'] for r in runs)}; temp_max {max(r['temp_max'] for r in runs)}")
print("rounds per arm:", collections.Counter(r["arm"] for r in runs))
for k, desc in ratios:
    v = list(RT[k].values()); vs = list(RTS[k].values())
    small = [RT[k][(t, n)] for t in types for n in Ns if n <= 8 and (t, n) in RT[k]]
    large = [RT[k][(t, n)] for t in types for n in Ns if n >= 16 and (t, n) in RT[k]]
    print(f"{k:15s} {desc:40s} all {min(v):.3f}-{max(v):.3f} | N<=8 {min(small):.3f}-{max(small):.3f} | N>=16 {min(large):.3f}-{max(large):.3f} | strict {min(vs) if vs else float('nan'):.3f}-{max(vs) if vs else float('nan'):.3f}")
worst = max((100 * (x[2] - x[1]) / x[0], k) for k, x in M.items())
print("worst spread %.2f%% at" % worst[0], worst[1])
