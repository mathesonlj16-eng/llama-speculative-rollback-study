"""Tests whether concurrent two-slot token differences follow request admission timing rather than the flag.

Both arms send the paired-checkpoint requests of the flag-off control. "flagoff" is the CUDA target prototype
build with its switch unset. The second request of the first pair is sent DELAY seconds after the first, and
later pairs are sent as in evaluate.py. Each record keeps the admission gap read from the server log.
"""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

here = Path(sys.argv[1]).resolve()
cap = sys.argv[2]
out = here / "admission-control-results"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
src = (here / "evaluate.py").read_text()
pair = "fs = [(a + j, ex.submit(complete, toks[a + j], j)) for j in range(2)]"
staggered = ("fs = [(a, ex.submit(complete, toks[a], 0))]; time.sleep(DELAY if a == 0 else 0); "
             "fs.append((a + 1, ex.submit(complete, toks[a + 1], 1)))")
assert src.count("\n\nmain()\n") == 1 and src.count(pair) == 1
code = compile(src.replace("\n\nmain()\n", "\n").replace(pair, staggered), "evaluate.py", "exec")
BASES = {"paired": Path(os.environ["STUDY_BASE"]), "flagoff": Path(os.environ["STUDY_FLAGOFF"])}
DELAYS = tuple(int(x) / 1000 for x in os.environ.get("ADMISSION_DELAYS_MS", "0,21,100").split(","))
REPEATS = int(os.environ.get("ADMISSION_REPEATS", "3"))


def load(base, delay):
    ns = {"__file__": str(here / "evaluate.py"), "__name__": "control"}
    sys.argv = ["evaluate.py", str(out), cap]
    exec(code, ns)
    ns["BASE"], ns["DELAY"] = base, delay
    return ns


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def first_gap_ms(name):
    stamps = re.findall(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I slot launch_slot_", (out / f"{name}.log").read_text(errors="replace"), re.M)
    (m0, s0, ms0, us0), (m1, s1, ms1, us1) = stamps[:2]
    t = lambda m, s, ms, us: ((int(m) * 60 + int(s)) * 1000 + int(ms)) + int(us) / 1000
    return t(m1, s1, ms1, us1) - t(m0, s0, ms0, us0)


out.mkdir()
rows = []
for delay in DELAYS:
    for rep in range(1, REPEATS + 1):
        for variant in (("paired", "flagoff") if rep % 2 else ("flagoff", "paired")):
            name = f"delay{int(delay * 1000)}-{variant}-r{rep}"
            rec = load(BASES[variant], delay)["run"](name, "paired", 2, 3)
            tokens = {str(x["prompt"]): x["tokens"] for x in rec["rows"]}
            rows.append(dict(delay_ms=int(delay * 1000), variant=variant, run=rep, ok=int(rec["ok"]),
                             first_pair_gap_ms=f"{first_gap_ms(name):.3f}", target_checkpoint_restores=rec["checkpoint_restores"],
                             token_arrays_sha256=digest(tokens),
                             prompt_sha256=" ".join(digest(tokens[k])[:12] for k in sorted(tokens, key=int))))

with (out / "admission-control.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for delay in DELAYS:
    group = [r for r in rows if r["delay_ms"] == int(delay * 1000)]
    sets = {}
    for r in group:
        sets.setdefault(r["token_arrays_sha256"], []).append(f"{r['variant']}-r{r['run']}")
    print(f"delay {int(delay * 1000)} ms: {sum(r['ok'] for r in group)}/{len(group)} ok, "
          f"{len(sets)} distinct token-array sets: " + "; ".join(f"{d[:12]} {v}" for d, v in sets.items()))
