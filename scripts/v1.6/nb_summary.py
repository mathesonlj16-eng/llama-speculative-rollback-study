# Summarize N=1..8 perf (default, bi-nb, bi-mmq) and bench-nb. Usage: python3 -I summarize.py <dir>
import re, sys, glob, os, csv, statistics as st
O = sys.argv[1]
SH = [(16, 2048), (512, 2048), (2048, 2048), (4096, 2048), (6144, 2048), (248320, 2048), (4096, 4096), (2048, 6144)]
pat = re.compile(r"MUL_MAT\(type_a=q8_0,type_b=f32,m=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\].*?-\s+([\d.]+) us/run")
d = {}
for f in glob.glob(os.path.join(O, "perf-*-r*.log")):
    arm = re.search(r"perf-(.*)-r\d+\.log", os.path.basename(f)).group(1)
    for line in open(f, errors="replace"):
        m = pat.search(line)
        if m and (int(m.group(1)), int(m.group(3))) in SH:
            d.setdefault((int(m.group(1)), int(m.group(3)), int(m.group(2))), {}).setdefault(arm, []).append(float(m.group(4)))
ns = sorted({k[2] for k in d})
med = lambda k, a: st.median(d[k][a])
for title, f in [("bi-nb / default", lambda k: med(k, "bi-nb")/med(k, "default")),
                 ("bi-mmq / default", lambda k: med(k, "bi-mmq")/med(k, "default")),
                 ("bi-nb us", lambda k: med(k, "bi-nb")), ("default us", lambda k: med(k, "default")),
                 ("bi-nb spread % (max-min)/med", lambda k: 100*(max(d[k]["bi-nb"])-min(d[k]["bi-nb"]))/med(k, "bi-nb"))]:
    print(f"\nper-op {title}; rows m x k, cols n")
    print(f"{'m':>7} {'k':>5} | " + " ".join(f"{n:>7}" for n in ns))
    for m, k in SH:
        print(f"{m:>7} {k:>5} | " + " ".join(f"{f((m,k,n)):7.2f}" if (m,k,n) in d else "      -" for n in ns))
b = {}
for r in csv.DictReader(open(os.path.join(O, "nb.csv"))):
    n = int(r["n"])
    if n < 0: continue
    b.setdefault((r["label"].rsplit("-r", 1)[0], n), []).append(float(r["ms_per_batch"]))
print("\nmodel 2B verify batch at depth 1024 (ms per batch, median [min-max])")
for n in sorted({k[1] for k in b}):
    a, c, e = b[("def", n)], b[("bi-nb", n)], b[("bi-mmq", n)]
    print(f"n={n:>2}: def {st.median(a):6.2f} [{min(a):.2f}-{max(a):.2f}] | bi-nb {st.median(c):6.2f} [{min(c):.2f}-{max(c):.2f}] {st.median(c)/st.median(a):.2f}x"
          f" | bi-mmq {st.median(e):6.2f} [{min(e):.2f}-{max(e):.2f}] {st.median(e)/st.median(a):.2f}x")
