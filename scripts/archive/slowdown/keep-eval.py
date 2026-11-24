"""Controlled test of the redundant draft restore and re-decode, plus a draft-length sweep."""
import collections
import csv
import gzip
import json
import shutil
import statistics as st
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
evaluate = here.parent / "target-ckpt-free" / "evaluate.py"
out = here / "keep-results"
prof_so, prof_dir = sys.argv[1], Path(sys.argv[2])
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
src = evaluate.read_text()
for old, new in [
    ("\n\nmain()\n", "\n"),
    ('here = Path(__file__).resolve().parent\nevidence = here.parent / "evidence"',
     f'here = Path(__file__).resolve().parent\nevidence = Path({str(here.parent / "evidence")!r})'),
    ('"prototype": PROTO / "bin"}[variant]', '"prototype": PROTO / "bin", "keep": PROTO / "bin-keep"}[variant]'),
    ('    if variant == "prototype":\n        env["LLAMA_SPEC_TARGET_CKPT_FREE"] = "1"\n',
     '    if variant in ("prototype", "keep"):\n        env["LLAMA_SPEC_TARGET_CKPT_FREE"] = "1"\n'
     '    if variant == "keep":\n        env["LLAMA_SPEC_DRAFT_KEEP"] = "1"\n'),
    ("    rec = dict(name=name,",
     '    env.pop("LLAMA_TRACE", None); env["LD_PRELOAD"] = CAP; env["Q35_PROF"] = f"{PROF_DIR}/{name}.events"\n'
     "    rec = dict(name=name,"),
    ("t0 = time.perf_counter()", 't0 = time.perf_counter(); rec["mono0"] = time.monotonic_ns()'),
    ('rec["wall_s"] = time.perf_counter() - t0', 'rec["wall_s"] = time.perf_counter() - t0; rec["mono1"] = time.monotonic_ns()'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ns = {"__file__": str(evaluate), "__name__": "keep", "PROF_DIR": str(prof_dir)}
sys.argv = ["evaluate.py", str(out), prof_so]
exec(compile(src, "evaluate.py", "exec"), ns)
out.mkdir()

plan = []
for rep in (1, 2, 3):
    for s in (1, 2):
        order = ["plain", "prototype", "keep"]
        plan += [(v, s, 3, rep) for v in (order if rep % 2 else order[::-1])]
for rep in (1, 2):
    for s in (1, 2):
        plan += [("keep", s, n, rep) for n in (1, 2, 5, 8)]

recs = []
for v, s, n, rep in plan:
    name = f"{v}-s{s}-n{n}-r{rep}"
    r = ns["run"](name, v, s, n)
    ev = prof_dir / f"{name}.events"
    if ev.exists():
        with ev.open("rb") as fi, gzip.open(out / f"{name}.events.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    recs.append(r)


def parts(r):
    c = collections.Counter()
    for line in gzip.open(out / f"{r['name']}.events.gz", "rt"):
        p = line.split()
        if len(p) == 6 and r["mono0"] <= int(p[3]) <= r["mono1"]:
            d = int(p[4]) / 1e6
            if p[1] in "PSL":
                c[p[0] + "_decode"] += d
            elif p[1] in "GR":
                c["checkpoint"] += d
    return c


groups = collections.defaultdict(list)
for r in recs:
    groups[(r["variant"], r["slots"], r["nmax"])].append(r)
fields = ["variant", "slots", "n_max", "runs", "ok", "median_wall_s", "tokens_per_s", "target_ms", "draft_ms", "checkpoint_ms", "acceptance", "rollback_failures"]
with (out / "summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for (v, s, n), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
        good = [r for r in rs if r["ok"]]
        if not good:
            w.writerow([v, s, n, len(rs), 0] + [""] * 7)
            continue
        pp = [parts(r) for r in good]
        wall = st.median(r["wall_s"] for r in good)
        dn = sum(y["timings"].get("draft_n", 0) for r in good for y in r["rows"])
        da = sum(y["timings"].get("draft_n_accepted", 0) for r in good for y in r["rows"])
        w.writerow([v, s, n, len(rs), len(good), f"{wall:.3f}", f"{288 / wall:.1f}", f"{st.median(x['T_decode'] for x in pp):.0f}",
                    f"{st.median(x['D_decode'] for x in pp):.0f}", f"{st.median(x['checkpoint'] for x in pp):.0f}",
                    f"{da / dn:.3f}" if dn else "", sum(r["rollback_failures"] for r in rs)])
print(open(out / "summary.csv").read())
print("failed:", [(r["name"], r["error"]) for r in recs if not r["ok"]])

seen = {}
for r in recs:
    if r["variant"] == "keep":
        for row in r["rows"]:
            seen.setdefault((tuple(row["prompt_tokens"]), tuple(row["tokens"])), []).append(r["name"])
with (out / "oracle-input.tsv").open("w") as f:
    for i, key in enumerate(seen):
        f.write(f"seq{i}\t{' '.join(map(str, key[0]))}\t{' '.join(map(str, key[1]))}\n")
(out / "runs.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in recs]))
