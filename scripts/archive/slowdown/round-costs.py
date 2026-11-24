"""Per-round cost breakdown from the raw profile events, for one-slot runs."""
import collections
import gzip
import json
import statistics as st
from pathlib import Path

res = Path(__file__).resolve().parent / "profile-results"
runs = {r["name"]: r for r in json.loads((res / "runs.json").read_text())}
EMITTED = 288


def load(name):
    r = runs[name]
    rows = []
    for line in gzip.open(res / f"{name}.events.gz", "rt"):
        p = line.split()
        if len(p) == 6 and r["mono0"] <= int(p[3]) <= r["mono1"]:
            rows.append((p[0], p[1], int(p[2]), int(p[3]), int(p[4]) / 1e6))
    return sorted(rows, key=lambda x: x[3])


for cfg in ("plain", "paired", "prototype"):
    vals = collections.defaultdict(list)
    for rep in (1, 2):
        rows = load(f"{cfg}-s1-r{rep}")
        c = collections.Counter()
        n = collections.Counter()
        for role, op, size, _, d in rows:
            if op in "GR":
                c[f"{role}_{op}_ms"] += d
                n[f"{role}_{op}_calls"] += 1
                c[f"{role}_{op}_MB"] += size / 1e6
            if op in "PSL":
                c[f"{role}_decode_ms"] += d
        # verification rounds are target decodes of 2 or more tokens after the prompt phase
        rounds = sum(1 for role, op, size, _, _ in rows if role == "T" and op == "P" and 2 <= size <= 4)
        c["rounds"] = rounds
        c.update(n)
        for k, v in c.items():
            vals[k].append(v)
    print(f"== {cfg}, one slot, median of 2 runs")
    for k in sorted(vals):
        print(f"  {k}: {st.median(vals[k]):.1f}")
    rounds = st.median(vals["rounds"]) if vals["rounds"] else 0
    if cfg != "plain" and rounds:
        ms = lambda k: st.median(vals[k]) if k in vals else 0.0
        ck = ms("D_G_ms") + ms("D_R_ms") + ms("T_G_ms") + ms("T_R_ms")
        print(f"  per round: target {ms('T_decode_ms') / rounds:.1f} ms, draft {ms('D_decode_ms') / rounds:.1f} ms, checkpoint {ck / rounds:.1f} ms, tokens {EMITTED / rounds:.2f}")
