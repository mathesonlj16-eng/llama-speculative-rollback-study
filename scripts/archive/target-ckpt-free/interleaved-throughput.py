"""Two-slot throughput with paired and prototype runs interleaved, five repetitions each."""
import json
import statistics as st
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "interleaved-results"
cap = sys.argv[1]
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
src = (here / "evaluate.py").read_text()
assert src.count("\n\nmain()\n") == 1
ns = {"__file__": str(here / "evaluate.py"), "__name__": "interleaved"}
sys.argv = ["evaluate.py", str(out), cap]
exec(compile(src.replace("\n\nmain()\n", "\n"), "evaluate.py", "exec"), ns)
out.mkdir()
recs = {"paired": [], "prototype": []}
for rep in range(1, 6):
    order = ("paired", "prototype") if rep % 2 else ("prototype", "paired")
    for v in order:
        recs[v].append(ns["run"](f"{v}-s2-n3-r{rep}", v, 2, 3))
summary = {}
for v, rs in recs.items():
    good = [r for r in rs if r["ok"]]
    walls = [r["wall_s"] for r in good]
    summary[v] = dict(ok=len(good), walls=[round(x, 3) for x in walls], median=st.median(walls), minimum=min(walls), maximum=max(walls),
                      rollbacks=[r["checkpoint_restores"] + r["snapshot_rollbacks"] for r in rs],
                      failures=sum(r["rollback_failures"] for r in rs),
                      distinct_outputs=len({json.dumps(sorted((x["prompt"], x["tokens"]) for x in r["rows"])) for r in good}))
    print(v, summary[v])
reduction = 100 * (1 - summary["prototype"]["median"] / summary["paired"]["median"])
summary["median_wall_reduction_percent"] = reduction
print(f"median wall reduction: {reduction:.1f}%")
(out / "summary.json").write_text(json.dumps(summary, indent=2))
