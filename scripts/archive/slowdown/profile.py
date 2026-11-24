"""Profiles plain, capacity-only, paired and prototype servers on the prototype evaluation workload."""
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
out = here / "profile-results"
prof_so, prof_dir = sys.argv[1], Path(sys.argv[2])
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")

src = evaluate.read_text()
for old, new in [
    ("\n\nmain()\n", "\n"),
    ('here = Path(__file__).resolve().parent\nevidence = here.parent / "evidence"',
     f'here = Path(__file__).resolve().parent\nevidence = Path({str(here.parent / "evidence")!r})'),
    ("    rec = dict(name=name,",
     '    env.pop("LLAMA_TRACE", None); env["LD_PRELOAD"] = CAP; env["Q35_PROF"] = f"{PROF_DIR}/{name}.events"\n'
     '    if name.startswith("plaincap"): env["Q35_TARGET_RS"] = "3"\n'
     "    rec = dict(name=name,"),
    ("t0 = time.perf_counter()", 't0 = time.perf_counter(); rec["mono0"] = time.monotonic_ns()'),
    ('rec["wall_s"] = time.perf_counter() - t0', 'rec["wall_s"] = time.perf_counter() - t0; rec["mono1"] = time.monotonic_ns()'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ns = {"__file__": str(evaluate), "__name__": "profile", "PROF_DIR": str(prof_dir)}
sys.argv = ["evaluate.py", str(out), prof_so]
exec(compile(src, "evaluate.py", "exec"), ns)
out.mkdir()

CONFIGS = [("plain", "plain"), ("plaincap", "plain"), ("paired", "paired"), ("prototype", "prototype")]
recs = []
for rep in (1, 2):
    for s in (1, 2):
        order = CONFIGS if rep == 1 else CONFIGS[::-1]
        for label, variant in order:
            name = f"{label}-s{s}-r{rep}"
            r = ns["run"](name, variant, s, 3)
            r["label"] = label
            recs.append(r)
            ev = prof_dir / f"{name}.events"
            with ev.open("rb") as fi, gzip.open(out / f"{name}.events.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)


def analyze(r):
    rows = []
    for line in gzip.open(out / f"{r['name']}.events.gz", "rt"):
        p = line.split()
        if len(p) == 6 and r["mono0"] <= int(p[3]) <= r["mono1"]:
            rows.append((p[0], p[1], int(p[2]), int(p[3]), int(p[4])))
    rows.sort(key=lambda x: x[3])
    ms = collections.Counter()
    tokens = collections.Counter()
    calls = collections.Counter()
    by_size = collections.defaultdict(list)
    open_batch = {}
    for role, op, n, _, d in rows:
        t = d / 1e6
        if op in "PSL":
            ms[f"{role}_decode"] += t
        elif op in "GR":
            ms["checkpoint"] += t
        else:
            ms["seq_rm"] += t
        if op == "P":
            tokens[role] += n
            calls[role] += 1
            if role in open_batch:
                by_size[(role, open_batch[role][0])].append(open_batch[role][1])
            open_batch[role] = [n, t]
        elif op in "SL" and role in open_batch:
            open_batch[role][1] += t
    for role, (n, t) in open_batch.items():
        by_size[(role, n)].append(t)
    busy, end = 0, 0
    for _, _, _, s, d in rows:
        a, b = max(s, end), s + d
        if b > a:
            busy += b - a
        end = max(end, b)
    return dict(wall=r["wall_s"] * 1000, ms=ms, tokens=tokens, calls=calls, by_size=by_size, busy=busy / 1e6,
                overlap=sum(ms.values()) - busy / 1e6)


res = {r["name"]: analyze(r) for r in recs if r["ok"]}
print("failed:", [r["name"] for r in recs if not r["ok"]])
fields = ["config", "slots", "wall_ms", "target_decode_ms", "draft_decode_ms", "checkpoint_ms", "seq_rm_ms", "unattributed_ms",
          "target_tokens", "target_calls", "draft_tokens", "draft_calls"]
table = []
for label, _ in CONFIGS:
    for s in (1, 2):
        names = [n for n in res if n.startswith(f"{label}-s{s}-")]
        med = lambda f: st.median(f(res[n]) for n in names)
        table.append(dict(config=label, slots=s, wall_ms=med(lambda x: x["wall"]),
                          target_decode_ms=med(lambda x: x["ms"]["T_decode"]), draft_decode_ms=med(lambda x: x["ms"]["D_decode"]),
                          checkpoint_ms=med(lambda x: x["ms"]["checkpoint"]), seq_rm_ms=med(lambda x: x["ms"]["seq_rm"]),
                          unattributed_ms=med(lambda x: x["wall"] - x["busy"]), target_tokens=med(lambda x: x["tokens"]["T"]),
                          target_calls=med(lambda x: x["calls"]["T"]), draft_tokens=med(lambda x: x["tokens"]["D"]),
                          draft_calls=med(lambda x: x["calls"]["D"])))
with (out / "components.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    for row in table:
        w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
print(open(out / "components.csv").read())
print("max overlapped ms in any run:", round(max(x["overlap"] for x in res.values()), 1))

with (out / "batch-latency.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "slots", "role", "batch_tokens", "count", "median_ms", "median_ms_per_token"])
    for label, _ in CONFIGS:
        for s in (1, 2):
            pool = collections.defaultdict(list)
            for n in res:
                if n.startswith(f"{label}-s{s}-"):
                    for k, v in res[n]["by_size"].items():
                        pool[k] += v
            for (role, size), v in sorted(pool.items()):
                if size > 0:
                    m = st.median(v)
                    w.writerow([label, s, role, size, len(v), round(m, 2), round(m / size, 2)])
print(open(out / "batch-latency.csv").read())
(out / "runs.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in recs]))
