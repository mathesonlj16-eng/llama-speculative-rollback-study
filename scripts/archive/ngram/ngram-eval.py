"""Compares plain decoding with model-free n-gram speculation on ordinary and repetitive workloads."""
import collections
import concurrent.futures as cf
import csv
import http.client
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "results"
cap_so = sys.argv[1]
evidence = here.parent / "evidence"
LIB = Path(os.environ["STUDY_BASE"]) / "fixed"
PORT = 18099

ORDINARY = [
    "The first twelve prime numbers are",
    "A computer science textbook explains sorting as follows:",
    "The capital cities of Europe include",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "In photosynthesis, plants convert sunlight into",
    "Once upon a time, in a small village by the sea,",
]
passage = ("The harbor office opens at seven each morning, and the clerk records every arriving vessel in a green ledger. "
           "Fishing boats unload along the north pier, while ferries use the south pier until the evening tide turns. ")
REPETITIVE = [
    "[\n" + "".join(f'  {{"id": {i}, "status": "active", "region": "north", "retries": 0, "owner": "team-a"}},\n' for i in range(1, 9)),
    "".join(f"def get_field_{i}(record):\n    if record is None:\n        return None\n    return record.get('field_{i}', None)\n\n" for i in range(1, 7)),
    f"Text: {passage}\nExact copy: {passage}\nText: {passage}\nExact copy:",
    "date,store,item,quantity,price\n" + "".join(f"2026-01-{d:02d},store-12,widget,5,19.99\n" for d in range(1, 13)),
    "".join(f"service_{i}:\n  image: registry.local/app:1.4\n  replicas: 2\n  restart: always\n  port: {8000 + i}\n" for i in range(1, 7)),
    "".join(f"12:00:{s:02d} INFO worker-3 request completed status=200 duration=14ms path=/api/items\n" for s in range(10, 22)),
]
WORKLOADS = {"ordinary": (ORDINARY, 48), "repetitive": (REPETITIVE, 96)}


def flags(kind, n, m, hits=1):
    return ["--spec-type", f"ngram-{kind}", f"--spec-ngram-{kind}-size-n", str(n), f"--spec-ngram-{kind}-size-m", str(m),
            f"--spec-ngram-{kind}-min-hits", str(hits)]


MOD = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "6", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "8"]
CONFIGS = {
    "plain": ([], 0),
    "simple-default": (["--spec-type", "ngram-simple"], 0),
    "simple-4-8": (flags("simple", 4, 8), 0),
    "simple-4-8-rs8": (flags("simple", 4, 8), 8),
    "map-k-4-8": (flags("map-k", 4, 8), 0),
    "map-k4v-4-8": (flags("map-k4v", 4, 8), 0),
    "mod-6-8": (MOD, 0),
    "mod-6-8-rs8": (MOD, 8),
    "cache-8": (["--spec-type", "ngram-cache", "--spec-draft-n-max", "8"], 0),
}


def request(method, path, body=None, timeout=300):
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


def complete(tokens, slot, n_predict):
    d = request("POST", "/completion", dict(prompt=tokens, id_slot=slot, temperature=0, seed=7, n_predict=n_predict,
                                            cache_prompt=False, return_tokens=True, ignore_eos=True))
    if len(d["tokens"]) != n_predict:
        raise RuntimeError("short completion")
    return d


def run(name, workload, config, slots):
    log_path = out / f"{name}.log"
    if log_path.exists():
        raise SystemExit(f"refusing to overwrite {log_path}")
    prompts, n_predict = WORKLOADS[workload]
    extra, rs = CONFIGS[config]
    cmd = json.loads((evidence / "baseline-command.json").read_text())
    cmd[0] = str(LIB / "llama-server")
    for k, v in (("-np", slots), ("-c", 1024 * slots), ("--port", PORT)):
        cmd[cmd.index(k) + 1] = str(v)
    cmd += ["-b", "2048", "-ub", "512", "-t", "8", "-tb", "8"] + extra
    env = dict(os.environ, LD_LIBRARY_PATH=str(LIB))
    for k in ("LD_PRELOAD", "LLAMA_TRACE", "Q35_TARGET_RS", "LLAMA_SPEC_TARGET_CKPT_FREE", "LLAMA_SPEC_DRAFT_KEEP"):
        env.pop(k, None)
    if rs:
        env.update(LD_PRELOAD=cap_so, Q35_TARGET_RS=str(rs))
    rec = dict(name=name, workload=workload, config=config, slots=slots, ok=False, error=None, rows=[])
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
            toks = [request("POST", "/tokenize", dict(content=p))["tokens"] for p in prompts]
            t0 = time.perf_counter()
            if slots == 1:
                rows = [(i, complete(t, 0, n_predict)) for i, t in enumerate(toks)]
            else:
                rows = []
                with cf.ThreadPoolExecutor(2) as ex:
                    for a in range(0, len(toks), 2):
                        fs = [(a + j, ex.submit(complete, toks[a + j], j, n_predict)) for j in range(2)]
                        rows += [(i, f.result()) for i, f in fs]
            rec["wall_s"] = time.perf_counter() - t0
            rec["rows"] = [dict(prompt=i, prompt_tokens=toks[i], tokens=d["tokens"], timings=d.get("timings", {})) for i, d in rows]
            rec["ok"] = True
        except Exception as e:  # a crash is a result
            rec["error"] = repr(e)
        finally:
            if proc.poll() is None:
                proc.terminate()
            rec["returncode"] = proc.wait()
    (out / f"{name}.json").write_text(json.dumps(rec))
    print(name, "ok" if rec["ok"] else f"FAILED {rec['error']}", f"wall={rec.get('wall_s', 0):.2f}s", flush=True)
    return rec


def main():
    out.mkdir(exist_ok=False)
    names = list(CONFIGS)
    recs = []
    for rep in (1, 2):
        for workload in WORKLOADS:
            for slots in (1, 2):
                for config in (names if rep == 1 else names[::-1]):
                    recs.append(run(f"{workload}-{config}-s{slots}-r{rep}", workload, config, slots))

    seen = {}
    for r in recs:
        for row in r["rows"]:
            seen.setdefault((tuple(row["prompt_tokens"]), tuple(row["tokens"])), set()).add(f"{r['workload']}|{r['config']}")
    with (out / "oracle-input.tsv").open("w") as f:
        for i, key in enumerate(seen):
            f.write(f"seq{i}\t{' '.join(map(str, key[0]))}\t{' '.join(map(str, key[1]))}\n")
    (out / "oracle-map.json").write_text(json.dumps({f"seq{i}": sorted(v) for i, v in enumerate(seen.values())}))

    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["workload"], r["slots"], r["config"])].append(r)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "slots", "config", "runs", "ok", "median_wall_s", "tokens_per_s", "speed_vs_plain", "draft_tokens", "accepted", "acceptance", "accepted_share_of_output"])
        for workload, (prompts, n_predict) in WORKLOADS.items():
            for slots in (1, 2):
                base = st.median(r["wall_s"] for r in groups[(workload, slots, "plain")] if r["ok"])
                for config in names:
                    rs = groups[(workload, slots, config)]
                    good = [r for r in rs if r["ok"]]
                    if not good:
                        w.writerow([workload, slots, config, len(rs), 0] + [""] * 7)
                        continue
                    wall = st.median(r["wall_s"] for r in good)
                    dn = sum(y["timings"].get("draft_n", 0) for r in good for y in r["rows"])
                    da = sum(y["timings"].get("draft_n_accepted", 0) for r in good for y in r["rows"])
                    total = n_predict * len(prompts)
                    w.writerow([workload, slots, config, len(rs), len(good), f"{wall:.3f}", f"{total / wall:.1f}", f"{base / wall:.3f}", dn, da,
                                f"{da / dn:.3f}" if dn else "", f"{da / (total * len(good)):.3f}"])
    print(open(out / "summary.csv").read())
    print("failed:", [(r["name"], r["error"]) for r in recs if not r["ok"]])
    (out / "runs.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in recs]))


main()
