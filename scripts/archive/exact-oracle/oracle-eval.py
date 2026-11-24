"""Bit-exact speculation oracle. Every server runs on the CPU reference kernels with the GPU hidden, where batch shape
does not change results. Each speculative configuration must reproduce plain decoding token for token."""
import concurrent.futures as cf
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
evidence = here.parent / "evidence"
out = here / "results"
USEREF, CAP = sys.argv[1], sys.argv[2]
PORT = 18100
BASE = Path(os.environ["STUDY_BASE"])
PROTO = Path(os.environ["STUDY_PROTO"])

PROMPTS = [
    ("The first twelve prime numbers are", 32),
    ("A computer science textbook explains sorting as follows:", 32),
    ("def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n", 32),
    ("Once upon a time, in a small village by the sea,", 32),
    ("".join(f"def get_field_{i}(record):\n    if record is None:\n        return None\n    return record.get('field_{i}', None)\n\n" for i in range(1, 5)), 48),
    ("date,store,item,quantity,price\n" + "".join(f"2026-01-{d:02d},store-12,widget,5,19.99\n" for d in range(1, 9)), 48),
]


def ngram(kind, n, m):
    return ["--spec-type", f"ngram-{kind}", f"--spec-ngram-{kind}-size-n", str(n), f"--spec-ngram-{kind}-size-m", str(m), f"--spec-ngram-{kind}-min-hits", "1"]


MOD = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "6", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "8"]
# name: (command file, library dir, extra args, target rollback capacity, extra environment)
CONFIGS = {
    "plain": ("baseline", BASE / "bin", [], 0, {}),
    "draft-stock": ("spec", BASE / "bin", [], 0, {}),
    "draft-paired-rs3": ("spec", BASE / "fixed", [], 3, {}),
    "draft-prototype": ("spec", PROTO / "bin", [], 0, {"LLAMA_SPEC_TARGET_CKPT_FREE": "1"}),
    "draft-keep": ("spec", PROTO / "bin-keep", [], 0, {"LLAMA_SPEC_TARGET_CKPT_FREE": "1", "LLAMA_SPEC_DRAFT_KEEP": "1"}),
    "ngram-simple": ("baseline", BASE / "fixed", ngram("simple", 4, 8), 0, {}),
    "ngram-simple-rs8": ("baseline", BASE / "fixed", ngram("simple", 4, 8), 8, {}),
    "ngram-mod": ("baseline", BASE / "fixed", MOD, 0, {}),
}


def request(method, path, body=None, timeout=1800):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path} returned {r.status}: {data[:300]!r}")
        return json.loads(data)
    finally:
        c.close()


def complete(tokens, slot, n):
    d = request("POST", "/completion", dict(prompt=tokens, id_slot=slot, temperature=0, seed=7, n_predict=n,
                                            cache_prompt=False, return_tokens=True, ignore_eos=True))
    if len(d["tokens"]) != n:
        raise RuntimeError("short completion")
    return d


def run(name, config, slots):
    log_path = out / f"{name}.log"
    if log_path.exists():
        raise SystemExit(f"refusing to overwrite {log_path}")
    kind, lib, extra, rs, extra_env = CONFIGS[config]
    cmd = json.loads((evidence / f"{'baseline' if kind == 'baseline' else 'spec'}-command.json").read_text())
    cmd[0] = str(lib / "llama-server")
    for k, v in (("-np", slots), ("-c", 1024 * slots), ("--port", PORT), ("-ngl", 0), ("-ngld", 0)):
        if k in cmd:
            cmd[cmd.index(k) + 1] = str(v)
    cmd += ["-b", "2048", "-ub", "512", "-t", "8", "-tb", "8"] + extra
    env = dict(os.environ, LD_LIBRARY_PATH=str(lib), CUDA_VISIBLE_DEVICES="", LD_PRELOAD=USEREF, LLAMA_TRACE="1")
    for k in ("Q35_TARGET_RS", "LLAMA_SPEC_TARGET_CKPT_FREE", "LLAMA_SPEC_DRAFT_KEEP", "LLAMA32776_TARGET_ROLLBACK"):
        env.pop(k, None)
    env.update(extra_env)
    if rs:
        env.update(LD_PRELOAD=f"{USEREF} {CAP}", Q35_TARGET_RS=str(rs))
    rec = dict(name=name, config=config, slots=slots, ok=False, error=None, rows=[])
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        try:
            for _ in range(1200):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}")
                try:
                    request("GET", "/health", timeout=2)
                    break
                except (OSError, RuntimeError, http.client.HTTPException):
                    time.sleep(0.25)
            else:
                raise RuntimeError("server start timeout")
            toks = [request("POST", "/tokenize", dict(content=p))["tokens"] for p, _ in PROMPTS]
            t0 = time.perf_counter()
            if slots == 1:
                rows = [(i, complete(t, 0, PROMPTS[i][1])) for i, t in enumerate(toks)]
            else:
                rows = []
                with cf.ThreadPoolExecutor(2) as ex:
                    for a in range(0, len(toks), 2):
                        fs = [(a + j, ex.submit(complete, toks[a + j], j, PROMPTS[a + j][1])) for j in range(2)]
                        rows += [(i, f.result()) for i, f in fs]
            rec["wall_s"] = time.perf_counter() - t0
            rec["rows"] = [dict(prompt=i, tokens=d["tokens"], timings=d.get("timings", {})) for i, d in rows]
            rec["ok"] = True
        except Exception as e:  # a crash is a result
            rec["error"] = repr(e)
        finally:
            if proc.poll() is None:
                proc.terminate()
            rec["returncode"] = proc.wait()
    text = log_path.read_text(errors="replace")
    rec["useref"] = text.count("USEREF applied")
    rec["cuda_mentions"] = text.count("CUDA0")
    rec["rollbacks"] = text.count("(snapshot rollback)") + text.count("(restore checkpoint)")
    rec["draft_tokens"] = sum(r["timings"].get("draft_n", 0) for r in rec["rows"])
    rec["accepted"] = sum(r["timings"].get("draft_n_accepted", 0) for r in rec["rows"])
    (out / f"{name}.json").write_text(json.dumps(rec))
    print(name, "ok" if rec["ok"] else f"FAILED {rec['error']}", f"wall={rec.get('wall_s', 0):.1f}s", f"useref={rec['useref']}",
          f"cuda={rec['cuda_mentions']}", f"draft={rec['draft_tokens']}", f"accepted={rec['accepted']}", f"rollbacks={rec['rollbacks']}", flush=True)
    return rec


def main():
    out.mkdir(exist_ok=False)
    recs = {}
    for config in CONFIGS:
        for slots in (1, 2):
            recs[(config, slots)] = run(f"{config}-s{slots}", config, slots)
    recs[("plain", "repeat")] = run("plain-s1-repeat", "plain", 1)
    ref = recs[("plain", 1)]
    if not ref["ok"]:
        raise SystemExit("reference run failed")
    want = {r["prompt"]: r["tokens"] for r in ref["rows"]}
    report = []
    print("\n== exact match against plain one-slot output")
    for (config, slots), r in recs.items():
        if not r["ok"]:
            line = f"{config} slots={slots}: FAILED {r['error']}"
        else:
            got = {x["prompt"]: x["tokens"] for x in r["rows"]}
            bad = []
            for p, w in want.items():
                g = got[p]
                if g != w:
                    i = next(k for k in range(len(w)) if g[k] != w[k])
                    bad.append(f"prompt {p} first diff at {i}: plain {w[i]} vs {g[i]}")
            line = f"{config} slots={slots}: {len(want) - len(bad)}/{len(want)} identical" + ("" if not bad else " | " + "; ".join(bad))
        print(line)
        report.append(line)
    (out / "report.txt").write_text("\n".join(report) + "\n")
    for key, record in recs.items():
        got = {row["prompt"]: row["tokens"] for row in record["rows"]}
        if not record["ok"] or got != want or not record["useref"] or record["cuda_mentions"]:
            raise SystemExit(f"Exact CPU oracle failed at {key}")


main()
