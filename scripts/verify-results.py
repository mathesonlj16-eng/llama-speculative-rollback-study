"""Recheck headline aggregates and comparisons from compact archived evidence."""
import csv
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parents[1] / "results"


def csv_rows(name):
    with (root / name).open(newline="") as f:
        return list(csv.DictReader(f))


def load(name):
    return json.loads((root / name).read_text())


runs = csv_rows("draft-runs.csv")
assert len(runs) == 34 and all(int(r["rollback_failures"]) == 0 for r in runs)
sweep = csv_rows("draft-sweep.csv")
for reported in sweep:
    group = [r for r in runs if (r["variant"], r["slots"], r["draft_max"]) ==
             (reported["variant"], reported["slots"], reported["n_max"])]
    assert len(group) == int(reported["runs"]) == int(reported["ok"])
    wall = median(float(r["wall_s"]) for r in group)
    assert abs(wall - float(reported["median_wall_s"])) < .00051
    assert abs(288 / wall - float(reported["tokens_per_s"])) < .051
    for field, output in [("target_call_ms", "target_ms"), ("draft_call_ms", "draft_ms"),
                          ("checkpoint_call_ms", "checkpoint_ms")]:
        assert abs(median(float(r[field]) for r in group) - float(reported[output])) <= .51
    if reported["variant"] == "keep":
        plain = next(r for r in sweep if r["variant"] == "plain" and r["slots"] == reported["slots"])
        assert float(plain["target_ms"]) - float(reported["target_ms"]) < float(reported["draft_ms"])
        assert float(plain["tokens_per_s"]) > float(reported["tokens_per_s"])

for mode, expected in [("baseline", 64.76), ("spec", 30.02)]:
    requests = load(f"bench_{mode}.json")
    assert len(requests) == 5
    assert round(median(x["predicted_per_second"] for x in requests), 2) == expected
    if mode == "spec":
        assert all((x["draft_n_accepted"], x["draft_n"]) == (90, 139) for x in requests)

def tokens(name):
    doc = load(f"oracle/{name}.json")
    assert doc["ok"], name
    return [row["tokens"] for row in doc["rows"]]

plain = tokens("results/plain-s1")
assert [len(row) for row in plain] == [32, 32, 32, 32, 48, 48]
for file in sorted((root / "oracle/results").glob("*.json")):
    assert tokens(f"results/{file.stem}") == plain, file.name
for slot in (1, 2):
    assert tokens(f"keep-checks/A-keepbin-keepoff-s{slot}") == tokens(f"results/draft-prototype-s{slot}") == plain
    assert tokens(f"keep-checks/C-keep-sample-s{slot}") == tokens(f"keep-checks/C-keepoff-sample-s{slot}")
assert tokens("keep-checks/A-keepbin-noflags-s1") == plain
cache1 = tokens("keep-checks/B-keep-cache-s1")
cache2 = tokens("keep-checks-b2/B2-draft-keep-cache-s2")
assert len(cache1) == 9 and cache1 == tokens("keep-checks/B-plain-cache-s1")
assert len(cache2) == 6 and cache2 == tokens("keep-checks-b2/B2-plain-cache-s2")
sampled = tokens("keep-checks/C-plain-sample-s1")
assert sampled == tokens("keep-checks/C-plain-sample-s1-repeat")
assert sum(a != b for a, b in zip(sampled, plain)) == 3
for mode, file in [("quantized", "rounded"), ("exact", "exact")]:
    raw = csv_rows(f"g7-sweep-{file}.csv")
    assert all(r["bit_equal"] == "1" if not float(r["size"]) else r["hits"] == "1" for r in raw)
    for point in (r for r in csv_rows("perturbation.csv") if r["activation"] == mode):
        match = [r for r in raw if (r["key"], r["index"], r["layer"], float(r["size"])) ==
                 (point["key"], point["index"], point["layer"], float(point["size"]))]
        assert len(match) == 8
        rms = float(format(median(float(r["rms_diff"]) for r in match), ".3g"))
        assert math.isclose(rms, float(point["median_output_rms"]), abs_tol=1.1e-6)
ngram = csv_rows("ngram.csv")
assert sum(int(r["ok"]) for r in ngram) == 72
assert max(float(r["tokens_per_s"]) for r in ngram if r["workload"] == "repetitive" and r["slots"] == "1"
           and r["config"] != "plain") == 55.8
comparisons = csv_rows("interleaved-runs.csv")
paired = [float(r["wall_s"]) for r in comparisons if r["variant"] == "paired"]
prototype = [float(r["wall_s"]) for r in comparisons if r["variant"] == "prototype"]
assert len(paired) == len(prototype) == 5
assert all(r["checkpoint_restores"] == "34" for r in comparisons if r["variant"] == "paired")
assert round(100 * (1 - median(prototype) / median(paired)), 1) == 15.3
for file in root.glob("g11-cpu-*.csv"):
    shape_rows = [r for r in csv_rows(file.name) if r["shape"] in ("whole", "chunk64")]
    assert len(shape_rows) == 384 and all(float(r["max_abs_diff_vs_singles"]) == 0 for r in shape_rows)
shape_rows = [r for r in csv_rows("g12-default.csv") if r["shape"] in ("whole", "chunk64")]
assert len(shape_rows) == 384
assert round(max(float(r["max_abs_diff_vs_singles"]) for r in shape_rows), 2) == 2.98
assert sum(r["top"] != r["ref_top"] for r in shape_rows) == 2

# 2026-11-24 rerun: CUDA flag-off control and forced admission-delay control
reruns = csv_rows("rerun-flag-off.csv")
forced = csv_rows("admission-control.csv")
assert len(reruns) == 12 and len(forced) == 22 and all(r["ok"] == "1" for r in reruns + forced)
assert sum(r["variant"] == "paired" for r in forced) == sum(r["variant"] == "flagoff" for r in forced) == 11
assert all(r["matches_archive"] == "1" for r in reruns if r["schedule"] == "serial")
assert len({r["token_arrays_sha256"] for r in reruns if r["schedule"] == "serial"}) == 1
common = {r["token_arrays_sha256"] for r in forced}
assert len(common) == 1
assert {r["target_checkpoint_restores"] for r in forced} == {"35", "37", "38", "39"}
odd = [r for r in reruns if r["schedule"] == "concurrent" and r["token_arrays_sha256"] not in common]
assert [(r["variant"], r["run"], r["target_checkpoint_restores"]) for r in odd] == [("flagoff", "r1", "38")]
assert any(abs(float(r["first_pair_gap_ms"]) - float(odd[0]["first_pair_gap_ms"])) < 3 and
           r["target_checkpoint_restores"] == "38" for r in forced)
print("Archived headline aggregates, CPU oracle comparisons, measured sweep points and rerun controls verified.")
