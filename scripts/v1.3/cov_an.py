import re, glob, os, collections, csv
X = os.path.join(os.environ.get("SM75_ROOT", os.path.expanduser("~/research/sm75-batch-invariance")), "evidence-p61/jfree-cov")
ansi = re.compile(r"\x1b\[[0-9;]*m")
line = re.compile(r"^(?:\[hb=([0-9a-f]+) hs=([0-9a-f]+)\])?\s*([A-Z_]+\(.*\)):\s+(OK|FAIL|not supported.*)$")
out = open(f"{X}/analysis.txt", "w")
def p(*a):
    print(*a); print(*a, file=out)
runs = {}
for f in sorted(glob.glob(f"{X}/*.log")):
    if f.endswith("log.txt"): continue
    cases = {}
    for l in open(f, errors="replace"):
        l = ansi.sub("", l).rstrip()
        m = line.match(l)
        if m:
            hb, hs, name, res = m.groups()
            cases[name] = (res.split()[0] if not res.startswith("not") else "unsupported", hb, hs)
    tag, b, nc = os.path.basename(f)[:-4].split(".")
    runs[(tag, b, nc)] = cases
    c = collections.Counter(v[0] for v in cases.values())
    ops = collections.Counter(n.split("(")[0] for n in cases)
    p(f"{tag:5} {b:12} {nc:9} cases={len(cases)} {dict(c)} ops={dict(ops)}")
rows = []
for (tag, b, nc), cases in runs.items():
    if nc != "nc0": continue
    pin = runs[(tag, b, "ncunset")]
    common = [n for n in cases if n in pin]
    hashed = [n for n in common if cases[n][1] and pin[n][1]]
    eqb = sum(cases[n][1] == pin[n][1] for n in hashed); eqs = sum(cases[n][2] == pin[n][2] for n in hashed)
    same_status = sum(cases[n][0] == pin[n][0] for n in common)
    p(f"compare {tag} {b}: common={len(common)} same_status={same_status} hashed={len(hashed)} batch_hash_eq={eqb} serial_hash_eq={eqs}")
    rows.append((tag, b, len(common), same_status, len(hashed), eqb, eqs))
    for n in common:
        if cases[n][0] == "FAIL" or pin[n][0] == "FAIL":
            p(f"   FAIL {tag} {b}: free={cases[n][0]} pinned={pin[n][0]} {n[:150]}")
with open(f"{X}/compare.csv", "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n"); w.writerow(["test", "build", "cases", "same_status", "hashed", "batch_hash_eq", "serial_hash_eq"]); w.writerows(rows)
