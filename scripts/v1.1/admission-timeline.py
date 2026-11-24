"""Cross-tabulates first-pair slot assignment and admission timing against the concurrent token outcome."""
import hashlib, json, re, sys
from pathlib import Path
LEG = {"437590e44d3a": "C-archive", "9170e15579ce": "C-rerun", "21efb4822d5c": "C-rerun-alt"}
def digest(t): return hashlib.sha256(json.dumps(t, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def events(log):
    t = lambda m, s, ms, us: ((int(m) * 60 + int(s)) * 1000 + int(ms)) + int(us) / 1000
    ev = []
    for m in re.finditer(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I slot (launch_slot_|  *operator\(\)): id +(\d) \| task (\d+) \| (.*)$", log, re.M):
        ev.append((t(*m.groups()[:4]), "launch" if "launch" in m.group(5) else "accept", int(m.group(6)), int(m.group(7))))
    return ev
rows = []
for res in [Path(p) for p in sys.argv[1:]]:
    for j in sorted(res.glob("*.json")):
        if j.name == "summary.json" or not (j.name.startswith("concurrent") or j.name.startswith("delay")): continue
        rec = json.loads(j.read_text())
        if not rec.get("rows"): continue
        toks = {str(x["prompt"]): x["tokens"] for x in rec["rows"]}
        ev = events(j.with_suffix(".log").read_text(errors="replace"))
        launches = [e for e in ev if e[1] == "launch"]
        l0, l1 = launches[0], launches[1]
        # verify steps completed by the first-admitted slot before the second request launched
        before = sum(1 for e in ev if e[1] == "accept" and e[2] == l0[2] and e[0] < l1[0])
        rows.append((res.parts[5] if res.parts[1] == "var" and "llama-v11" in res.parts else res.parts[-3], j.stem, f"slot{l0[2]}<-task{l0[3]}", round(l1[0] - l0[0], 1), before,
                     LEG.get(digest(toks)[:12], digest(toks)[:12]), rec["checkpoint_restores"]))
from collections import Counter
for r in rows: print(*r, sep="\t")
print("\n(first-slot, steps-before-join, outcome) counts:")
for k, c in sorted(Counter((r[2].split("<-")[0], min(r[4], 2), r[5]) for r in rows).items()): print(k, c)
