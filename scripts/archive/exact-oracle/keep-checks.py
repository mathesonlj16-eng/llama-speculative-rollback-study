"""Keep-draft checks on the CPU reference path, reusing oracle-eval.py.

A. Flag off: the keep-draft build without LLAMA_SPEC_DRAFT_KEEP must equal plain greedy output.
B. Prompt cache: repeated and extended prompts with cache_prompt on; keep-draft must equal plain with the same requests.
C. Sampling: temperature 0.8, fixed seed; keep-draft must equal the same build with keep-draft off.
"""
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
useref, cap = sys.argv[1:3]
src = (here / "oracle-eval.py").read_text()
assert src.count("\n\nmain()\n") == 1
ns = {"__file__": str(here / "oracle-eval.py"), "__name__": "keepchecks"}
sys.argv = ["oracle-eval.py", useref, cap]
exec(compile(src.replace("\n\nmain()\n", "\n"), "oracle-eval.py", "exec"), ns)
out = here / "keep-checks"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
out.mkdir()
ns["out"] = out
PROTO = ns["PROTO"]
ns["CONFIGS"].update({
    "keepbin-noflags": ("spec", PROTO / "bin-keep", [], 0, {}),
    "keepbin-keepoff": ("spec", PROTO / "bin-keep", [], 0, {"LLAMA_SPEC_TARGET_CKPT_FREE": "1"}),
})
request = ns["request"]
MODE = {"cache": False, "temp": 0.0}


def complete(tokens, slot, n):
    body = dict(prompt=tokens, id_slot=slot, seed=7, n_predict=n, cache_prompt=MODE["cache"], return_tokens=True, ignore_eos=True)
    body.update(temperature=MODE["temp"], top_k=40, top_p=0.95) if MODE["temp"] > 0 else body.update(temperature=0)
    d = request("POST", "/completion", body)
    if len(d["tokens"]) != n:
        raise RuntimeError("short completion")
    return d


ns["complete"] = complete
BASE_PROMPTS = list(ns["PROMPTS"])
CACHE_BASE = [BASE_PROMPTS[0], BASE_PROMPTS[1], BASE_PROMPTS[4]]
EXT = " In summary,"


def cache_prompts(slots):
    items = []
    for text, n in CACHE_BASE:
        items.append([(text, n), (text, n), (text + EXT, n)])
    if slots == 1:
        return [x for group in items for x in group]
    seq = []
    for k in range(3):
        for group in (items[0], items[1]):
            seq.append(group[k])
    return seq + items[2]  # third prompt group runs in pairs on both slots too


def tokens(rec):
    return {r["prompt"]: r["tokens"] for r in rec["rows"]}


def run(name, config, slots, cache=False, temp=0.0, prompts=None):
    MODE.update(cache=cache, temp=temp)
    ns["PROMPTS"] = prompts or BASE_PROMPTS
    rec = ns["run"](name, config, slots)
    rec["prompts"] = ns["PROMPTS"]
    (out / f"{name}.json").write_text(json.dumps(rec))
    return rec


plain = {r["prompt"]: r["tokens"] for r in json.loads((here / "results" / "plain-s1.json").read_text())["rows"]}
proto = {r["prompt"]: r["tokens"] for r in json.loads((here / "results" / "draft-prototype-s1.json").read_text())["rows"]}
lines = []


def compare(label, got, want):
    if not got:
        lines.append(f"{label}: run failed")
        return False
    diffs = [p for p in want if got.get(p) != want[p]]
    lines.append(f"{label}: {len(want) - len(diffs)}/{len(want)} identical" + (f" (differ: {diffs})" if diffs else ""))
    return not diffs


# A. flag off
for slots in (1, 2):
    r = run(f"A-keepbin-keepoff-s{slots}", "keepbin-keepoff", slots)
    compare(f"A keep build, keep flag off, slots={slots} vs plain greedy", tokens(r) if r["ok"] else {}, plain)
    compare(f"A keep build, keep flag off, slots={slots} vs prototype greedy", tokens(r) if r["ok"] else {}, proto)
r = run("A-keepbin-noflags-s1", "keepbin-noflags", 1)
compare("A keep build, no flags, slots=1 vs plain greedy", tokens(r) if r["ok"] else {}, plain)

# B. prompt cache
for slots in (1, 2):
    pr = cache_prompts(slots)
    ref = run(f"B-plain-cache-s{slots}", "plain", slots, cache=True, prompts=pr)
    kp = run(f"B-keep-cache-s{slots}", "draft-keep", slots, cache=True, prompts=pr)
    compare(f"B keep vs plain, prompt cache on, slots={slots}", tokens(kp) if kp["ok"] else {}, tokens(ref) if ref["ok"] else {})
    if ref["ok"]:
        reused = [r["timings"].get("prompt_n") for r in sorted(kp["rows"], key=lambda x: x["prompt"])] if kp["ok"] else []
        lines.append(f"B slots={slots} prompt tokens evaluated per request (keep run): {reused}")
        first = {i: t for i, t in tokens(ref).items() if pr[i] in BASE_PROMPTS}
        want = {i: plain[BASE_PROMPTS.index(pr[i])] for i in first}
        compare(f"B plain with cache vs plain without cache (unextended prompts), slots={slots}", first, want)

# C. sampling
for slots in (1, 2):
    off = run(f"C-keepoff-sample-s{slots}", "keepbin-keepoff", slots, temp=0.8)
    on = run(f"C-keep-sample-s{slots}", "draft-keep", slots, temp=0.8)
    compare(f"C keep on vs keep off, temperature 0.8 seed 7, slots={slots}", tokens(on) if on["ok"] else {}, tokens(off) if off["ok"] else {})
p1 = run("C-plain-sample-s1", "plain", 1, temp=0.8)
p2 = run("C-plain-sample-s1-repeat", "plain", 1, temp=0.8)
compare("C control: plain sampled repeat", tokens(p2) if p2["ok"] else {}, tokens(p1) if p1["ok"] else {})
if p1["ok"]:
    lines.append(f"C sampling active: plain sampled output differs from greedy on {sum(tokens(p1)[i] != plain[i] for i in plain)}/6 prompts")
    off1 = json.loads((out / "C-keepoff-sample-s1.json").read_text())
    if off1["ok"]:
        lines.append(f"C information only: speculative sampled vs plain sampled identical on {sum(tokens(off1)[i] == tokens(p1)[i] for i in plain)}/6 prompts")

print("\n== keep-draft checks")
print("\n".join(lines))
(out / "report.txt").write_text("\n".join(lines) + "\n")
