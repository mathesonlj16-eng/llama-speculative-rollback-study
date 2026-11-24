"""Runs plain, paired-checkpoint and prototype servers on one fixed workload and records raw results."""
import concurrent.futures as cf
import csv
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
evidence = here.parent / "evidence"
out = Path(sys.argv[1])
BASE = Path(os.environ["STUDY_BASE"])
PROTO = Path(os.environ["STUDY_PROTO"])
CAP = sys.argv[2]
PORT = 18098
N_PREDICT = 48
PROMPTS = [
    "The first twelve prime numbers are",
    "A computer science textbook explains sorting as follows:",
    "The capital cities of Europe include",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "In photosynthesis, plants convert sunlight into",
    "Once upon a time, in a small village by the sea,",
]


def request(method, path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path} returned {r.status}: {data[:300]!r}")
        return json.loads(data)
    finally:
        c.close()


def complete(tokens, slot):
    body = dict(prompt=tokens, id_slot=slot, temperature=0, seed=7, n_predict=N_PREDICT,
                cache_prompt=False, return_tokens=True, ignore_eos=True)
    d = request("POST", "/completion", body)
    if len(d["tokens"]) != N_PREDICT:
        raise RuntimeError("short completion")
    return d


def run(name, variant, slots, nmax):
    log_path = out / f"{name}.log"
    if log_path.exists():
        raise SystemExit(f"refusing to overwrite {log_path}")
    plain = variant == "plain"
    cmd = json.loads((evidence / ("baseline-command.json" if plain else "spec-command.json")).read_text())
    lib = {"plain": BASE / "bin", "paired": BASE / "fixed", "prototype": PROTO / "bin"}[variant]
    cmd[0] = str(lib / "llama-server")
    for k, v in (("-np", slots), ("-c", 1024 * slots), ("--port", PORT), ("--spec-draft-n-max", nmax)):
        if k in cmd:
            cmd[cmd.index(k) + 1] = str(v)
    cmd += ["-b", "2048", "-ub", "512", "-t", "8", "-tb", "8"]
    env = dict(os.environ, LD_LIBRARY_PATH=str(lib), LLAMA_TRACE="1")
    for k in ("LD_PRELOAD", "LLAMA32776_TARGET_ROLLBACK", "Q35_TARGET_RS", "LLAMA_SPEC_TARGET_CKPT_FREE"):
        env.pop(k, None)
    if variant == "paired":
        env.update(LD_PRELOAD=CAP, Q35_TARGET_RS=str(nmax))
    if variant == "prototype":
        env["LLAMA_SPEC_TARGET_CKPT_FREE"] = "1"

    rec = dict(name=name, variant=variant, slots=slots, nmax=nmax, ok=False, error=None, rows=[])
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        try:
            for _ in range(480):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}")
                try:
                    request("GET", "/health", timeout=2)
                    break
                except (OSError, RuntimeError, http.client.HTTPException):
                    time.sleep(0.25)
            else:
                raise RuntimeError("server start timeout")
            rec["gpu_mib"] = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
            toks = [request("POST", "/tokenize", dict(content=p))["tokens"] for p in PROMPTS]
            t0 = time.perf_counter()
            if slots == 1:
                rows = [(i, complete(t, 0)) for i, t in enumerate(toks)]
            else:
                rows = []
                with cf.ThreadPoolExecutor(2) as ex:
                    for a in range(0, len(toks), 2):
                        fs = [(a + j, ex.submit(complete, toks[a + j], j)) for j in range(2)]
                        rows += [(i, f.result()) for i, f in fs]
            rec["wall_s"] = time.perf_counter() - t0
            for i, d in rows:
                t = d.get("timings", {})
                rec["rows"].append(dict(prompt=i, prompt_tokens=toks[i], tokens=d["tokens"], timings=t))
            rec["ok"] = True
        except Exception as e:  # recorded as a result: a crash here is evidence
            rec["error"] = repr(e)
        finally:
            if proc.poll() is None:
                proc.terminate()
            rec["returncode"] = proc.wait()
    text = log_path.read_text(errors="replace")
    rec["snapshot_rollbacks"] = text.count("(snapshot rollback)")
    rec["checkpoint_restores"] = text.count("(restore checkpoint)")
    rec["rollback_failures"] = text.count("snapshot rollback failed")
    (out / f"{name}.json").write_text(json.dumps(rec))
    print(name, "ok" if rec["ok"] else f"FAILED {rec['error']}", f"wall={rec.get('wall_s', 0):.2f}s",
          f"snap={rec['snapshot_rollbacks']} ckpt={rec['checkpoint_restores']}", flush=True)
    return rec


def main():
    out.mkdir(parents=True, exist_ok=False)
    plan = [("plain", s, 3, 0) for s in (1, 2)]
    for rep in range(1, 4):
        for s in (1, 2):
            order = ("paired", "prototype") if rep % 2 else ("prototype", "paired")
            plan += [(v, s, 3, rep) for v in order]
    plan += [("prototype", 2, n, 1) for n in (1, 8)] + [("paired", 2, n, 1) for n in (1, 8)]
    recs = [run(f"{v}-s{s}-n{n}-r{r}", v, s, n) for v, s, n, r in plan]

    seen = {}
    for r in recs:
        for row in r["rows"]:
            key = (tuple(row["prompt_tokens"]), tuple(row["tokens"]))
            seen.setdefault(key, []).append(r["name"])
    with (out / "oracle-input.tsv").open("w") as f:
        for i, (key, names) in enumerate(seen.items()):
            f.write(f"seq{i}\t{' '.join(map(str, key[0]))}\t{' '.join(map(str, key[1]))}\n")
    (out / "oracle-map.json").write_text(json.dumps(
        {f"seq{i}": names for i, names in enumerate(seen.values())}))

    groups = {}
    for r in recs:
        groups.setdefault((r["variant"], r["slots"], r["nmax"]), []).append(r)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "slots", "n_max", "runs", "ok", "median_wall_s", "median_tokens_per_s", "draft_tokens",
                    "draft_accepted", "acceptance", "snapshot_rollbacks", "checkpoint_restores", "rollback_failures", "gpu_mib_max"])
        for (v, s, n), rs in groups.items():
            good = [x for x in rs if x["ok"]]
            walls = [x["wall_s"] for x in good]
            total = N_PREDICT * len(PROMPTS)
            dn = sum(y["timings"].get("draft_n", 0) for x in good for y in x["rows"])
            da = sum(y["timings"].get("draft_n_accepted", 0) for x in good for y in x["rows"])
            w.writerow([v, s, n, len(rs), len(good), f"{statistics.median(walls):.3f}" if walls else "",
                        f"{total / statistics.median(walls):.2f}" if walls else "", dn, da, f"{da / dn:.3f}" if dn else "",
                        sum(x["snapshot_rollbacks"] for x in rs), sum(x["checkpoint_restores"] for x in rs),
                        sum(x["rollback_failures"] for x in rs), max((x.get("gpu_mib", 0) for x in rs), default=0)])


main()
