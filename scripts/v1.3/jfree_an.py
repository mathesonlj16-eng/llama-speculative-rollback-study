import re, glob, os, csv, statistics, collections
ROOT = os.environ.get("SM75_ROOT", os.path.expanduser("~/research/sm75-batch-invariance"))
X = f"{ROOT}/evidence-p61/jfree"
out = open(f"{X}/analysis.txt", "w")
def p(*a):
    print(*a); print(*a, file=out)

# 1. which J was used per setting
for f in sorted(glob.glob(f"{X}/jbest.*.raw")):
    js = sorted({(int(a), int(b)) for a, b in re.findall(r"ne11=(\d+) inv=1 J_best=(\d+)", open(f).read())})
    p("J", os.path.basename(f), " ".join(f"N{a}->J{b}" for a, b in js))

# 2. hashes: within run hb==hs is the test; across settings compare hb for same case (seeded)
pat = re.compile(r"\[hb=([0-9a-f]+) hs=([0-9a-f]+)\]\s+MUL_MAT\(([^)]*)\)")
def load(f):
    return {c: (hb, hs) for hb, hs, c in pat.findall(open(f).read())}
rows = []
for b in ("pool-d4f-pr", "pool-s75-pr"):
    for grid in ("big", "small"):
        runs = {nc: load(f"{X}/{b}.nc{nc}.{grid}.1234.log") for nc in ("unset", "0", "64")}
        base = runs["unset"]
        for nc in ("0", "64"):
            r = runs[nc]
            same_b = sum(r[c][0] == base[c][0] for c in base)
            same_s = sum(r[c][1] == base[c][1] for c in base)
            p(f"hash {b} {grid}: ncols={nc} vs pinned: batch-hash equal {same_b}/{len(base)}, serial-hash equal {same_s}/{len(base)}")
            rows.append((b, grid, nc, len(base), same_b, same_s))
        inv = sum(v[0] == v[1] for v in base.values())
        p(f"hash {b} {grid}: pinned run batch==serial {inv}/{len(base)}")
# old pool run (before the edit) vs today's unset: same binary behaviour?
old = load(f"{ROOT}/evidence-p61/pr-pool/pool-d4f-pr.1234.log")
new = load(f"{X}/pool-d4f-pr.ncunset.big.1234.log")
p(f"hash pool-d4f-pr big: today's pinned vs 2026-11-24 pool run: equal {sum(new[c][0]==old[c][0] for c in old)}/{len(old)}")
# cross-path: dp4a vs MMA outputs (expected different)
a = load(f"{X}/pool-d4f-pr.ncunset.big.1234.log"); b2 = load(f"{X}/pool-s75-pr.ncunset.big.1234.log")
p(f"hash big: dp4a vs MMA path pinned, equal {sum(a[c][0]==b2[c][0] for c in a)}/{len(a)}")
with open(f"{X}/hash-compare.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["build", "grid", "ncols", "cases", "batch_hash_equal_to_pinned", "serial_hash_equal_to_pinned"]); w.writerows(rows)

# 3. perf: parse test-backend-ops perf lines
pp = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+)[^)]*\):\s+\d+ runs -\s+([\d.]+) us/run")
obs = collections.defaultdict(list)
for f in glob.glob(f"{X}/perf/*.s[123].log"):
    arm = os.path.basename(f).split(".")[0]
    for t, m, n, k, us in pp.findall(open(f).read()):
        obs[(arm, t, int(n))].append(float(us))
types = sorted({k[1] for k in obs}); ns = sorted({k[2] for k in obs})
with open(f"{X}/perf-medians.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["type", "N", "def_us", "pin_us", "free_us", "n_obs", "pin_over_def", "free_over_def", "free_over_pin", "spread_pct_max"])
    for t in types:
        for n in ns:
            med = {a: statistics.median(obs[(a, t, n)]) for a in ("def", "pin", "free")}
            spr = max((max(obs[(a, t, n)]) - min(obs[(a, t, n)])) / med[a] * 100 for a in med)
            w.writerow([t, n] + [round(med[a], 2) for a in ("def", "pin", "free")] + [min(len(obs[(a, t, n)]) for a in med),
                       round(med["pin"]/med["def"], 3), round(med["free"]/med["def"], 3), round(med["free"]/med["pin"], 3), round(spr, 2)])
p(open(f"{X}/perf-medians.csv").read())
