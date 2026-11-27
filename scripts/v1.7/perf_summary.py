# Per-op perf summary for nbkq: median us/run over 5 rounds per arm; ratios vs default. Usage: python -I summarize.py <logdir>
import glob, os, re, statistics as st, sys
D = sys.argv[1]
pat = re.compile(r"MUL_MAT\(type_a=(\w+),type_b=f32,m=(\d+),n=(\d+),k=(\d+),.*?([\d.]+) us/run")
t = {}
for f in glob.glob(os.path.join(D, "perf-*-r*.log")):
    arm = re.match(r"perf-(.+)-r\d+\.log", os.path.basename(f)).group(1)
    for line in open(f, errors="replace"):
        m = pat.search(line)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(4)), int(m.group(3)))
            t.setdefault((arm,) + key, []).append(float(m.group(5)))
def med(a, k): return st.median(t[(a,) + k])
def spread(a, k): v = t[(a,) + k]; return (max(v) - min(v)) / st.median(v)
keys = sorted({k[1:] for k in t})
print("type m k n | default us | new/def | mmq/def | new spread")
worst = {}
for k in keys:
    d, nw, mq = med("default", k), med("bi-new", k), med("bi-mmq", k)
    print(f"{k[0]} {k[1]} {k[2]} {k[3]} | {d:8.2f} | {nw/d:5.2f} | {mq/d:5.2f} | {spread('bi-new', k)*100:4.1f}%")
    if k[3] >= 2:
        w = worst.setdefault((k[0], k[1] == 16), [9, 0]); w[0] = min(w[0], nw/d); w[1] = max(w[1], nw/d)
for (ty, small), (lo, hi) in sorted(worst.items()):
    print(f"range N=2..8 new/default {ty} {'16-row' if small else 'real shapes'}: {lo:.2f}-{hi:.2f}")
print("max spread any arm:", f"{max(spread(a, k) for a in ('default', 'bi-new', 'bi-mmq') for k in keys)*100:.1f}%")
