import re, os, csv, collections
X = os.path.join(os.environ.get("SM75_ROOT", os.path.expanduser("~/research/sm75-batch-invariance")), "evidence-p61/jfree-moe2")
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"\[hb=([0-9a-f]+) hs=([0-9a-f]+)\]\s+(MUL_MAT_ID\([^)]*\)):\s+(\S+)")
def load(f):
    d = {}
    for l in open(f, errors="replace"):
        m = pat.search(ansi.sub("", l))
        if m: d[m.group(3)] = (m.group(1), m.group(2), m.group(4))
    return d
out = open(f"{X}/analysis.txt", "w")
def p(*a): print(*a); print(*a, file=out)
for nc in ("0", "unset"):
    js = sorted({int(j) for j in re.findall(r"J_best=(\d+)", open(f"{X}/jbest.{nc}.txt").read())})
    pairs = sorted({(int(a), int(b)) for a, b in re.findall(r"ne11=(\d+) inv=1 J_best=(\d+)", open(f"{X}/jbest.{nc}.txt").read())})
    p(f"J used ncols={nc}: {js}")
rows = []
for b in ("pool-d4f-pr", "pool-s75-pr"):
    a, r, f = (load(f"{X}/{b}.nc{x}.log") for x in ("unset.a", "unset.b", "0.a"))
    st = collections.Counter(v[2] for v in f.values())
    inv = sum(v[0] == v[1] for v in f.values())
    rep = sum(a[n][0] == r[n][0] and a[n][1] == r[n][1] for n in a)
    fp = sum(f[n][0] == a[n][0] and f[n][1] == a[n][1] for n in a)
    p(f"{b}: cases={len(f)} status(free)={dict(st)} free batch==serial {inv}/{len(f)} | pinned A==B {rep}/{len(a)} | free==pinned {fp}/{len(a)}")
    rows.append((b, len(f), dict(st).get("OK", 0), inv, rep, fp))
    diff = [n for n in a if not (f[n][0] == a[n][0])]
    for n in diff[:6]: p("   differs:", n)
with open(f"{X}/compare.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["build", "cases", "ok_free", "free_batch_eq_serial", "pinned_repeat_eq", "free_eq_pinned"]); w.writerows(rows)
