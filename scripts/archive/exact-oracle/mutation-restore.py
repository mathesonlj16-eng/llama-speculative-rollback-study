"""Positive control: runs selected configurations with rollbacks deliberately broken and checks the oracle flags them."""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
useref, cap, brk = sys.argv[1:4]
src = (here / "oracle-eval.py").read_text()
for old, new in [
    ("\n\nmain()\n", "\n"),
    ('out = here / "results"', 'out = here / "mutation2-results"'),
    ("    rec = dict(name=name, config=config", '    env["LD_PRELOAD"] = env["LD_PRELOAD"] + " " + BREAK\n    rec = dict(name=name, config=config'),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
ns = {"__file__": str(here / "oracle-eval.py"), "__name__": "mutation", "BREAK": brk}
sys.argv = ["oracle-eval.py", useref, cap]
exec(compile(src, "oracle-eval.py", "exec"), ns)
out = ns["out"]
out.mkdir(exist_ok=False)
want = {r["prompt"]: r["tokens"] for r in json.loads((here / "results" / "plain-s1.json").read_text())["rows"]}
lines = []
for config in ("draft-stock", "draft-paired-rs3", "ngram-simple", "ngram-mod"):
    r = ns["run"](f"{config}-s1-broken", config, 1)
    log = (out / f"{config}-s1-broken.log").read_text(errors="replace")
    stats = [l for l in log.splitlines() if "BREAK_RESTORE" in l]
    if not r["ok"]:
        lines.append(f"{config}: run failed ({r['error']}) {stats}")
        continue
    got = {x["prompt"]: x["tokens"] for x in r["rows"]}
    differing = sum(got[p] != w for p, w in want.items())
    lines.append(f"{config}: {differing}/{len(want)} prompts differ from plain | {stats[-1] if stats else 'no BREAK report'}")
print("\n== mutation results (each must show differences)")
print("\n".join(lines))
(out / "report.txt").write_text("\n".join(lines) + "\n")
