import re, os, collections
X = os.path.join(os.environ.get("SM75_ROOT", os.path.expanduser("~/research/sm75-batch-invariance")), "evidence-p61/jfree-cov")
ansi = re.compile(r"\x1b\[[0-9;]*m")
line = re.compile(r"^(?:\[hb=([0-9a-f]+) hs=([0-9a-f]+)\])?\s*([A-Z_]+\(.*\)):\s+(OK|FAIL.*|not supported.*)$")
def load(f):
    d = {}
    for l in open(f, errors="replace"):
        m = line.match(ansi.sub("", l).rstrip())
        if m: d[m.group(3)] = (m.group(4), m.group(1), m.group(2))
    return d
for b in ("pool-d4f-pr", "pool-s75-pr"):
    fr, pn = load(f"{X}/suite.{b}.nc0.log"), load(f"{X}/suite.{b}.ncunset.log")
    diff = [n for n in fr if n in pn and fr[n][1] != pn[n][1]]
    print(b, "hash-differs by op:", dict(collections.Counter(n.split("(")[0] for n in diff)))
    for n in diff[:4]: print("   ", n[:170], "free hb==hs:", fr[n][1] == fr[n][2], "pin hb==hs:", pn[n][1] == pn[n][2])
    only = [n for n in fr if n not in pn] + [n for n in pn if n not in fr]
    print(b, "cases in only one run:", len(only), [n[:90] for n in only[:3]])
for nc in ("nc0", "ncunset"):
    f = f"{X}/suite.pool-d4f-pr.{nc}.log"
    txt = [ansi.sub("", l).rstrip() for l in open(f, errors="replace")]
    fails = [l for l in txt if "FAIL" in l]
    print(f"--- d4f {nc}: {len(fails)} FAIL lines; first 4:"); [print("   ", l[:200]) for l in fails[:4]]
    print("    last lines:", [l[:160] for l in txt[-6:]])
