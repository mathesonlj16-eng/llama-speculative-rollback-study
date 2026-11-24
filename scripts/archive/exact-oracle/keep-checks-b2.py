"""Rerun of check B at two slots. The first attempt built an odd-length request list, so pairing failed.
Each slot gets one prompt three times: as is, repeated, and extended. Cache must be on and reused."""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
useref, cap = sys.argv[1:3]
src = (here / "oracle-eval.py").read_text()
assert src.count("\n\nmain()\n") == 1
ns = {"__file__": str(here / "oracle-eval.py"), "__name__": "keepchecks_b2"}
sys.argv = ["oracle-eval.py", useref, cap]
exec(compile(src.replace("\n\nmain()\n", "\n"), "oracle-eval.py", "exec"), ns)
out = here / "keep-checks-b2"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
out.mkdir()
ns["out"] = out
request = ns["request"]


def complete(tokens, slot, n):
    d = request("POST", "/completion", dict(prompt=tokens, id_slot=slot, seed=7, n_predict=n, temperature=0, cache_prompt=True,
                                            return_tokens=True, ignore_eos=True))
    if len(d["tokens"]) != n:
        raise RuntimeError("short completion")
    return d


ns["complete"] = complete
P = ns["PROMPTS"]
a, b = P[0], P[4]
ext = lambda p: (p[0] + " In summary,", p[1])
ns["PROMPTS"] = [a, b, a, b, ext(a), ext(b)]
lines = []
recs = {}
for config in ("plain", "draft-keep"):
    recs[config] = ns["run"](f"B2-{config}-cache-s2", config, 2)
for config, r in recs.items():
    lines.append(f"{config}: ok={r['ok']} error={r['error']} prompt tokens evaluated per request="
                 f"{[x['timings'].get('prompt_n') for x in sorted(r['rows'], key=lambda x: x['prompt'])]}")
if all(r["ok"] for r in recs.values()):
    p = {x["prompt"]: x["tokens"] for x in recs["plain"]["rows"]}
    k = {x["prompt"]: x["tokens"] for x in recs["draft-keep"]["rows"]}
    lines.append(f"B keep vs plain, prompt cache on, slots=2: {sum(p[i] == k[i] for i in p)}/{len(p)} identical")
    s1 = json.loads((here / "keep-checks" / "B-plain-cache-s1.json").read_text())
    by_text = {s1["prompts"][x["prompt"]][0]: x["tokens"] for x in s1["rows"]}
    same = sum(by_text.get(ns["PROMPTS"][i][0]) == p[i] for i in p)
    lines.append(f"B plain two-slot cached vs plain one-slot cached, same prompt text: {same}/{len(p)} identical")
print("\n".join(lines))
(out / "report.txt").write_text("\n".join(lines) + "\n")
