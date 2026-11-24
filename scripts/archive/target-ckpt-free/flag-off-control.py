"""Separates admission-order effects from code changes in the flag-off comparison.

serial     : two slots, one request at a time, so no admission race exists.
concurrent : repeated two-slot runs, with admission order read from each server log.
"""
import json
import re
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "flag-off-control-results"
cap = sys.argv[1]
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
src = (here / "evaluate.py").read_text()
assert src.count("\n\nmain()\n") == 1 and src.count("cf.ThreadPoolExecutor(2)") == 1
BASES = {"paired": Path(os.environ["STUDY_BASE"]), "flagoff": Path(os.environ["STUDY_FLAGOFF"])}


def load(workers, base):
    ns = {"__file__": str(here / "evaluate.py"), "__name__": "control"}
    sys.argv = ["evaluate.py", str(out), cap]
    code = src.replace("\n\nmain()\n", "\n").replace("cf.ThreadPoolExecutor(2)", f"cf.ThreadPoolExecutor({workers})")
    exec(compile(code, "evaluate.py", "exec"), ns)
    ns["BASE"] = base
    return ns


def tokens(rec):
    return {x["prompt"]: x["tokens"] for x in rec["rows"]}


def admission(name):
    text = (out / f"{name}.log").read_text(errors="replace")
    return "".join(re.findall(r"launch_slot_.*?id +(\d+) \|", text))


out.mkdir()
results = {}
for mode, workers in (("serial", 1), ("concurrent", 2)):
    for variant, base in BASES.items():
        for rep in range(1, 4):
            name = f"{mode}-{variant}-r{rep}"
            rec = load(workers, base)["run"](name, "paired", 2, 3)
            results[name] = dict(tokens=tokens(rec), admission=admission(name), restores=rec["checkpoint_restores"], ok=rec["ok"])

for mode in ("serial", "concurrent"):
    print(f"\n== {mode}")
    for name, r in results.items():
        if name.startswith(mode):
            print(name, "ok" if r["ok"] else "FAILED", "restores", r["restores"], "admission", r["admission"])
    ref = results[f"{mode}-paired-r1"]["tokens"]
    for name, r in results.items():
        if name.startswith(mode):
            print(f"{name} identical to paired-r1: {sum(r['tokens'].get(k) == ref[k] for k in ref)}/{len(ref)}")
    groups = {}
    for name, r in results.items():
        if name.startswith(mode):
            groups.setdefault(r["admission"], []).append(name)
    for order, names in groups.items():
        base = results[names[0]]["tokens"]
        same = all(results[n]["tokens"] == base for n in names)
        print(f"admission {order}: {len(names)} runs, variants {sorted({n.split('-')[1] for n in names})}, all tokens identical = {same}")
(out / "summary.json").write_text(json.dumps(results))
