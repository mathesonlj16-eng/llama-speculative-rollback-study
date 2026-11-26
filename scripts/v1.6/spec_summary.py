# Timing/acceptance summary for spec-bi CSVs. Usage: python3 -I summarize.py <dir>
import csv, glob, os, sys, statistics as st
O = sys.argv[1]
agg = {}  # (arm, prefix, k) -> list of per-run dicts
for f in sorted(glob.glob(os.path.join(O, "*.csv"))):
    if os.path.basename(f) == "gpu.csv": continue
    rows = list(csv.DictReader(open(f)))
    if not rows: continue
    lab = rows[0]["label"]; arm = lab.split("-", 1)[1] if lab.startswith("r") else lab
    per = {}
    for r in rows:
        k = int(r["k"]); x = per.setdefault(k, {"ms": 0, "tok": 0, "acc": 0, "dr": 0, "rb": 0, "rounds": 0, "t": [0, 0, 0, 0]})
        x["ms"] += float(r["ms"]); x["tok"] += int(r["n_gen"]); x["acc"] += int(r["accepted"]); x["dr"] += int(r["drafted"])
        x["rb"] += int(r["redecode_bad"]); x["rounds"] += int(r["rounds"])
        for i, c in enumerate(["ms_draft", "ms_verify", "ms_ckpt", "ms_rollback"]): x["t"][i] += float(r[c])
    for k, x in per.items(): agg.setdefault((arm, int(rows[0]["prefix"]), k), []).append(x)
print("tok/s = generated tokens / decode time (prefill excluded); median over runs [min-max]; speed-up vs plain (k=0) same arm")
print("acc = accepted/drafted; per-round ms: draft / verify / ckpt / rollback (median run)")
for arm, pre in sorted({(a, p) for a, p, _ in agg}):
    base = st.median([x["tok"]*1000/x["ms"] for x in agg[(arm, pre, 0)]])
    print(f"\n== {arm} prefix={pre}  runs={len(agg[(arm, pre, 0)])}")
    for k in sorted({k for a, p, k in agg if a == arm and p == pre}):
        xs = agg[(arm, pre, k)]; ts = [x["tok"]*1000/x["ms"] for x in xs]; m = sorted(xs, key=lambda x: x["ms"])[len(xs)//2]
        line = f"k={k}: {st.median(ts):6.1f} tok/s [{min(ts):.1f}-{max(ts):.1f}]  x{st.median(ts)/base:.2f}"
        if k > 0:
            R = m["rounds"]
            line += f"  acc {m['acc']/m['dr']:.2f}  tok/round {m['tok']/R:.2f}  per-round ms {m['t'][0]/R:.1f} / {m['t'][1]/R:.1f} / {m['t'][2]/R:.2f} / {m['t'][3]/R:.2f}  redecode_bad {sum(x['rb'] for x in xs)}"
        print(line)
