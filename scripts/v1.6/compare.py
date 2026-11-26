# Compare speculative vs plain (k = 0) token/logit-hash streams within each .tok file, and plain streams across files.
# Usage: python3 -I compare.py file.tok [file.tok ...]
import sys
def load(f):
    d = {}
    for line in open(f):
        a = line.split(); k, s = int(a[0]), int(a[1])
        d[(k, s)] = [tuple(x.split(":")) for x in a[2:]]
    return d
def first_diff(x, y, i):  # i = 0 token, 1 hash
    for j, (u, v) in enumerate(zip(x, y)):
        if u[i] != v[i]: return j
    return None
D = {f: load(f) for f in sys.argv[1:]}
for f, d in D.items():
    ks = sorted({k for k, _ in d}); ss = sorted({s for _, s in d})
    print(f"== {f}")
    for k in ks:
        if k == 0: continue
        tok_bad = [s for s in ss if first_diff(d[(0, s)], d[(k, s)], 0) is not None]
        hash_bad = [s for s in ss if first_diff(d[(0, s)], d[(k, s)], 1) is not None]
        fh = [first_diff(d[(0, s)], d[(k, s)], 1) for s in hash_bad]
        print(f"k={k}: seqs with token diff {len(tok_bad)}/{len(ss)}; seqs with logit-hash diff {len(hash_bad)}/{len(ss)}"
              + (f"; first hash diff at positions {fh}" if fh else ""))
fs = list(D)
for i in range(len(fs)):
    for j in range(i + 1, len(fs)):
        a, b = D[fs[i]], D[fs[j]]
        ss = sorted({s for k, s in a if k == 0} & {s for k, s in b if k == 0})
        hb = sum(first_diff(a[(0, s)], b[(0, s)], 1) is not None for s in ss)
        tb = sum(first_diff(a[(0, s)], b[(0, s)], 0) is not None for s in ss)
        print(f"plain {fs[i]} vs {fs[j]}: token diff {tb}/{len(ss)}, hash diff {hb}/{len(ss)}")
