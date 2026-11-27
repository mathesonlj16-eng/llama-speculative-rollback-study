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
# v1.1.0: toolkit, target-capacity and admission-order controls
C_ARCHIVE, C_RERUN = "437590e44d3a", "9170e15579ce"
fm = csv_rows("v1.1/flagoff-matrix.csv")
assert len(fm) == 120 and all(r["ok"] == "1" for r in fm)
assert {r["tokens"] for r in fm if r["run"].startswith("serial")} == {"S-archive"}
for r in fm:
    if r["run"].startswith("concurrent"):
        if r["target_n_rs_seq"] == "0":   # archive and no-op shim arms on every build
            assert r["token_arrays_sha256"].startswith(C_ARCHIVE) and r["restores"] == "34", r
        else:                             # capacity 3: default set, or the alternative set after prompt 1 led
            assert r["tokens"] in ("C-rerun", "C-rerun-alt"), r
            assert (r["tokens"] == "C-rerun-alt") == (r["admission"][0] == "1" and float(r["first_pair_gap_ms"]) >= 15), r
assert sum(r["target_n_rs_seq"] == "0" and r["run"].startswith("concurrent") and r["arm"] != "archive-2026-11-23" for r in fm) == 18
rv = csv_rows("v1.1/reverse-admission-matrix.csv")
assert len(rv) == 90 and all(r["ok"] == "1" for r in rv)
for r in rv:
    if int(r["delay_ms"]) >= 10:
        assert r["tokens"] == ("C-rerun-alt" if r["arm"].endswith("-cap") else "new-6e86f030d727"), r
am = csv_rows("v1.1/admission-matrix.csv")
assert len(am) == 54 and all(r["ok"] == "1" and r["token_arrays_sha256"].startswith(C_RERUN) for r in am)
for name in ("logit-toolkit-and-shape.csv", "logit-capacity.csv"):
    for r in csv_rows(f"v1.1/{name}"):
        if r["comparison"].split(":")[0] in ("toolkit", "build", "repeat") or r["comparison"].endswith(":singles:rs0-vs-rs3"):
            assert r["bit_identical"] == r["positions"] == "288", r
        else:
            assert int(r["bit_identical"]) < 288, r
for r in csv_rows("v1.1/interleaved-matrix.csv"):
    p, q = float(r["paired_median_s"]), float(r["prototype_median_s"])
    assert abs(100 * (1 - q / p) - float(r["reduction_percent"])) < .06 and r["failures"] == "0"
    assert set(r["paired_restores"].split()) == ({"34"} if r["arm"].endswith("nocap") else {"39"})
audit = [r for r in csv_rows("v1.1/archive-capacity-audit.csv") if r["slots"] == "2" and
         ("flag-off-control-results/concurrent" in r["log"] or "interleaved-results/paired" in r["log"] or "eval-results/paired-s2-n3" in r["log"])]
assert len(audit) == 14 and all((r["target_n_rs_seq"], r["checkpoint_restores"]) in (("0", "34"), ("3", "39")) for r in audit)
# v1.2.0: dp4a (Pascal-style) MMQ supplement
rounds = csv_rows("v1.2/rounds.csv")
med = csv_rows("v1.2/medians.csv")
arms = ["s75m0", "s75m2", "s75pron", "fm0", "fm2", "fpron", "nm0"]
assert len(med) == 5 * 12
for r in med:
    for a in arms:
        obs = [float(x["us"]) for x in rounds if (x["arm"], x["type"], x["N"]) == (a, r["type"], r["N"])]
        assert len(obs) == int(r[f"{a}_n"]) == 10, (a, r["type"], r["N"])
        assert abs(median(obs) - float(r[f"{a}_us"])) < .006, (a, r["type"], r["N"])
mi = {(r["type"], r["N"]): r for r in med}
for r in csv_rows("v1.2/ratios.csv"):
    for col in ("fpron/fm2", "s75pron/s75m2", "fpron/s75pron", "fm0/s75m0", "fm2/s75m2", "fm2/fm0", "nm0/fm0"):
        a, b = col.split("/")
        m = mi[(r["type"], r["N"])]
        q = float(m[f"{a}_us"]) / float(m[f"{b}_us"])  # medians.csv is rounded to 0.01 us
        assert abs(q - float(r[col])) < 3e-4 * q + 6e-5, (col, r["type"], r["N"])
    assert .99 < float(r["fm2/s75m2"]) < 1.011                 # chunked-MMVQ control
    if int(r["N"]) >= 16:
        assert 1.06 <= float(r["s75pron/s75m2"]) <= 1.471 and .629 <= float(r["fpron/fm2"]) <= 1.113
        assert float(r["fm0/s75m0"]) < .56
assert len(csv_rows("v1.2/run-clocks.csv")) >= 70
corr = csv_rows("v1.2/correctness.csv")
cells = {"small": 300, "bigN": 450}
for r in corr:
    if r["arm"] in ("fm2", "pool-d4f-pr", "pool-s75-pr") or (r["arm"] == "fpron" and r["grid"] == "small"):
        assert (r["exit"], r["fail"], int(r["ok"])) == ("0", "0", cells[r["grid"]]), r
    elif r["arm"] == "fpron":   # timed binary's harness: context pool exhausted, not a mismatch
        assert r["exit"] == "134" and r["fail"] == "0" and r["aborted"] == "1", r
    else:                       # default and fixed-warp modes are not batch-invariant
        assert r["exit"] == "1" and int(r["fail"]) > 0, r
sc = csv_rows("v1.2/static-check-dp4a.csv")
assert [(r["build"], r["mmq_IMMA"], r["mmq_IDP.4A"]) for r in sc] == [("fixed-warp", "290683", "0"), ("d4f-fw", "0", "650496"), ("d4f-pr", "0", "650496")]
st = {r["build"]: r for r in csv_rows("v1.2/static-check.csv")}
assert st["fixed-warp"]["mmq_IMMA"] == st["pr"]["mmq_IMMA"] == "290683" and all(st[b]["all_IMMA"] == st[b]["all_HMMA"] == "0" for b in ("d4f-fw", "d4f-pr", "d4-fw"))
# v1.3.0: tile-width pin
c3 = csv_rows("v1.3/correctness.csv")
assert len(c3) == 16 and all(r["exit"] == "0" and r["fail"] == "0" and int(r["ok"]) == (450 if r["grid"] == "big" else 300) for r in c3)
hc = csv_rows("v1.3/hash-compare.csv")
assert len(hc) == 8 and all(r["batch_hash_equal_to_pinned"] == r["serial_hash_equal_to_pinned"] == r["cases"] for r in hc)
for r in csv_rows("v1.3/perf-medians.csv"):
    d, pn, f = float(r["def_us"]), float(r["pin_us"]), float(r["free_us"])
    assert abs(f / d - float(r["free_over_def"])) < .0015 and abs(pn / d - float(r["pin_over_def"])) < .0015
    if r["N"] == "512":
        assert .99 <= f / d <= 1.004 and pn / d > 2.3
# v1.3.1: coverage and MoE
cc = {(r["test"], r["build"]): r for r in csv_rows("v1.3/coverage-compare.csv")}
for t in ("types", "oddk"):
    for b in ("pool-d4f-pr", "pool-s75-pr"):
        r = cc[(t, b)]
        assert r["cases"] == r["same_status"] == r["batch_hash_eq"] == r["serial_hash_eq"], r
for r in csv_rows("v1.3/moe-compare.csv"):
    assert r["cases"] == r["ok_free"] == r["free_batch_eq_serial"] == r["pinned_repeat_eq"] == r["free_eq_pinned"] == "52", r
# v1.3.2: non-quantized MUL_MAT_ID gap
mg = csv_rows("v1.3/moe-gap.csv")
def st(v, t): return [r["status"] for r in mg if r["variant"] == v and r["type"] == t]
assert st("default", "bf16") == ["FAIL"] * 9 and st("default_after_patch", "bf16") == ["FAIL"] * 9
assert sorted(st("default", "f16")) == ["FAIL"] * 2 + ["OK"] * 7
assert all(r["n"] == "256" and r["m"] == "2048" for r in mg if r["variant"] == "default" and r["type"] == "f16" and r["status"] == "FAIL")
assert st("default", "f32") == st("default", "q8_0") == ["OK"] * 9
assert st("fixA_f32sort", "bf16") == ["OK"] * 9 and sorted(st("fixA_f32sort", "f16")) == ["FAIL"] * 2 + ["OK"] * 7
assert st("fixA_plus_B_nommf", "f16") == ["ABORT_needs_sync_assert"]
# v1.4.0: PR #32081 batch-invariant replay of the study fixtures
e1 = csv_rows("e1-cuda.csv")
e1off, e1on = csv_rows("v1.4/e1-flag-off.csv"), csv_rows("v1.4/e1-flag-on.csv")
strip = lambda rows: [{k: v for k, v in r.items() if k != "backend"} for r in rows]
assert strip(e1off) == strip(e1)   # flag off on the PR base reproduces the v1.0.0 archive
assert sum(r["bit_identical_to_ref"] == "0" for r in e1off) == 24 and {r["plan"] for r in e1off if r["top"] != e1off[0]["top"]} == {"split5", "split6", "split8"}
assert len(e1on) == 34 and all(r["bit_identical_to_ref"] == "1" and float(r["max_abs_diff_vs_ref"]) == 0 and r["top"] == e1on[0]["top"] for r in e1on)
seqs = csv_rows("v1.4/logit-cmp-by-sequence.csv")
tot = {}
for r in seqs:
    t = tot.setdefault((r["phase"], r["comparison"]), [0, 0, 0])
    t[0] += int(r["positions"]); t[1] += int(r["nonzero"]); t[2] += int(r["argmax_differs"])
assert len(tot) == 16 and all(v[0] == 1440 for v in tot.values())
assert all(tot[("a3", f"off-singles-vs-{s}")][1] == 1440 for s in ("whole", "c4", "c8", "pair4"))
assert [tot[("a3", f"on-singles-vs-{s}")][1:] for s in ("whole", "c4", "c8", "pair4")] == [[768, 0], [603, 0], [607, 0], [1051, 2]]
assert all(tot[("a4", f"on-singles-vs-{s}")][1:] == [0, 0] for s in ("whole", "c4", "c8")) and tot[("a4", "on-singles-vs-pair4")][1:] == [209, 0]
assert tot[("a5", "ctx1024-singles-vs-pair4")][1:] == [0, 0] and tot[("a5", "ctx2048-vs-ctx1024-singles")][1:] == [209, 0]
for r in seqs:   # 256-cell boundary: flag-on single-sequence residual only where the span exceeds 256 tokens
    if r["phase"] == "a3" and r["comparison"] == "on-singles-vs-whole":
        assert (int(r["nonzero"]) > 0) == (int(r["prompt_tokens"]) + int(r["gen_tokens"]) - 1 > 256), r
pair_a4 = {r["key"]: r for r in seqs if r["comparison"] == "on-singles-vs-pair4" and r["phase"] == "a4"}
ctx_a5 = {r["key"]: r for r in seqs if r["comparison"] == "ctx2048-vs-ctx1024-singles"}
assert all((pair_a4[k]["nonzero"], pair_a4[k]["first_nonzero_index"], pair_a4[k]["max_abs_diff"]) == (ctx_a5[k]["nonzero"], ctx_a5[k]["first_nonzero_index"], ctx_a5[k]["max_abs_diff"]) for k in pair_a4)
# v1.5.0: invariant-mode flash-attention KV-split fix
assert (root / "v1.5/e1-flag-off.csv").read_bytes() == (root / "v1.4/e1-flag-off.csv").read_bytes()
assert (root / "v1.5/e1-flag-on.csv").read_bytes() == (root / "v1.4/e1-flag-on.csv").read_bytes()
fa = csv_rows("v1.5/logit-cmp-by-sequence.csv")
fat = {}
for r in fa:
    t = fat.setdefault(r["comparison"], [0, 0, 0])
    t[0] += int(r["positions"]); t[1] += int(r["nonzero"]); t[2] += int(r["argmax_differs"])
assert len(fat) == 6 and all(v == [1440, 0, 0] for v in fat.values()), fat   # flag on: all shapes incl. pair4; default: unchanged
bt = [r for r in csv_rows("v1.5/bench-times.csv") if int(r["depth"]) > 0]
assert len(bt) == 60
def med(cfg, d, f):
    xs = sorted((d if f == "pp_ms" else int(r["tg_tokens"])) / float(r[f]) for r in bt if r["label"].rsplit("-r", 1)[0] == cfg and int(r["depth"]) == d)
    assert len(xs) == 5; return xs[2]
for d in (256, 1024, 4096):
    for f in ("pp_ms", "tg_ms"):
        assert .99 <= med("fafix-bi", d, f) / med("base-bi", d, f) <= 1.005          # patch cost in invariant mode
        assert .997 <= med("fafix-def", d, f) / med("base-def", d, f) <= 1.003       # default mode unaffected
    assert .30 <= med("base-bi", d, "tg_ms") / med("base-def", d, "tg_ms") <= .33     # invariant-mode decode cost
    assert .76 <= med("base-bi", d, "pp_ms") / med("base-def", d, "pp_ms") <= .80
# v1.5.1: operator-level padded-KV FA test
pk = csv_rows("v1.5/fa-padkv-test.csv")
assert len(pk) == 80 and all(r["status"] == "OK" for r in pk if r["build"] == "pr-head+fa-fix" or r["test"] == "pr_fa_batch_inv")
def crosses(r): p0, nb = int(r["p0"]), int(r["nb"]); return (p0 + nb + 255) // 256 != (p0 + 1 + 255) // 256
head = [r for r in pk if r["build"] == "pr-head" and r["test"] == "padkv"]
assert len(head) == 24 and sum(crosses(r) for r in head) == 16
assert all(r["status"] == "OK" for r in head if not crosses(r))      # controls pass on the PR head
fails = sorted((r["hs"], r["nh"], r["p0"], r["type_KV"]) for r in head if r["status"] == "FAIL")
assert fails == sorted([(h, n, p, t) for h, n in (("256", "2"),) for p in ("250", "255", "500") for t in ("f16", "q8_0")] + [("128", "8", "500", "f16")]), fails
assert all(crosses(r) and 0 < int(r["differ"]) < int(r["total"]) for r in head if r["status"] == "FAIL")
# v1.5.2: two-sequence (stream-count) FA test
sk = csv_rows("v1.5/fa-streams-test.csv")
assert len(sk) == 128 and all(r["status"] == "OK" for r in sk if r["build"] == "pr-head+fa-fix" or r["test"] == "pr_fa_batch_inv")
key = lambda r: (r["build"], r["hs"], r["nh"], r["p0"], r["nb"], r["type_KV"])
assert {key(r): r["status"] for r in sk if r["test"] == "padkv"} == {key(r): r["status"] for r in pk if r["test"] == "padkv"}   # v1.5.1 statuses reproduce
st_head = [r for r in sk if r["build"] == "pr-head" and r["test"] == "streams"]
assert len(st_head) == 24 and all(r["streams"] == "2" for r in st_head)
sfails = sorted((r["hs"], r["L"], r["nb"], r["type_KV"]) for r in st_head if r["status"] == "FAIL")
assert sfails == sorted([("256", "4096", nb, t) for nb in ("1", "4") for t in ("f16", "q8_0")] + [("128", L, nb, "q8_0") for L in ("1024", "4096") for nb in ("1", "4")]), sfails
assert all(r["status"] == "OK" for r in st_head if r["L"] == "256")
assert all(.88 <= int(r["differ"]) / int(r["total"]) <= .94 for r in st_head if r["status"] == "FAIL")
# v1.6.0: fast batch-invariant GEMV kernels (sm_75) and speculative decoding
def sweep_rows(name):   # sweep files repeat the header once per harness invocation
    rs = csv_rows(name); k = next(iter(rs[0])); return [r for r in rs if r[k] != k]
def cand(rs, c, q=None): return [r for r in rs if r["cand"] == c and (q is None or r["qtype"] == q)]
def failing(rs): return sum(int(r["bad_vs_ref"]) > 0 for r in rs)
q8h = sweep_rows("v1.6/q8-sweep-harness-vs-pr-bi.csv")
assert len(cand(q8h, "prod")) == 1240 and failing(cand(q8h, "prod")) == 0 and sum(int(r["fp64_bad"]) for r in q8h) == 0
assert sum(int(r["compared"]) for r in cand(q8h, "prod")) == 109981760
assert failing(cand(q8h, "ctrl_a_assoc")) == 1236
assert all(failing(cand(q8h, c)) == 1240 for c in ("ctrl_b_unfused", "ctrl_c_splitK2", "ctrl_d_mmvq_quant", "ctrl_e_reverse"))
q8n = sweep_rows("v1.6/q8-sweep-intree-n1-8.csv")   # every column of N = 1..8 (new kernels) vs N = 16 (forced MMQ)
assert sum(int(r["ref_selfinv_bad"]) for r in q8n) == 0 and failing(cand(q8n, "prod")) == 0
assert sum(int(r["compared"]) for r in cand(q8n, "prod")) == 85349888
INFORMATIVE = {"normal", "zeros", "cancel", "single", "mid"}
for name, comp in (("v1.6/kq-sweep-harness-vs-pr-bi.csv", 68640320), ("v1.6/kq-sweep-intree-n1.csv", 75962880)):
    kq = sweep_rows(name)
    assert sum(int(r["ref_selfinv_bad"]) for r in kq) == 0 and failing(cand(kq, "prod")) == 0 and sum(int(r["fp64_bad"]) for r in kq) == 0
    assert sum(int(r["compared"]) for r in cand(kq, "prod")) == comp
kq = sweep_rows("v1.6/kq-sweep-intree-n1.csv")
inf = lambda c, q: [r for r in cand(kq, c, q) if r["family"] in INFORMATIVE]
for c in ("ctrl_splitK2", "ctrl_fp32_dmA", "ctrl_m_first", "ctrl_mmvq_quant"):
    assert failing(inf(c, "q4_K")) == len(inf(c, "q4_K")) == 480, c
assert failing(inf("ctrl_assoc", "q4_K")) == 239 and failing(cand(kq, "ctrl_unfused_m", "q4_K")) == 0
for c in ("ctrl_splitK2", "ctrl_flat_fold", "ctrl_reverse", "ctrl_mmvq_quant"):
    assert failing(cand(kq, c, "q6_K")) == len(cand(kq, c, "q6_K")) == 306, c
assert failing(cand(kq, "ctrl_fp32_scale", "q6_K")) == 0
assert [r["family"] for r in cand(kq, "ctrl_tmp_fmul", "q6_K") if int(r["bad_vs_ref"]) > 0] == ["subnormal"] * 2
for name in ("v1.6/variants-q8.csv", "v1.6/variants-kq.csv"):
    assert all(r["bad"] == "0" for r in csv_rows(name))
p1 = [r for r in csv_rows("v1.6/perf-q8-n1.csv") if int(r["m"]) >= 512]
v3 = [float(r["bi-v3_med"]) / float(r["default_med"]) for r in p1]
assert .77 <= min(v3) and max(v3) <= .962 and all(2.6 <= float(r["bi-mmq_med"]) / float(r["default_med"]) <= 5.14 for r in p1)
pn = [r for r in csv_rows("v1.6/perf-q8-nb.csv") if int(r["m"]) >= 512 and 2 <= int(r["n"]) <= 8]
assert len(pn) == 42 and all(r["default_runs"] == r["bi-nb_runs"] == r["bi-mmq_runs"] == "5" for r in pn)
nb = [float(r["bi-nb_med"]) / float(r["default_med"]) for r in pn]
assert .68 <= min(nb) and max(nb) <= 1.012 and all(1.35 <= float(r["bi-mmq_med"]) / float(r["default_med"]) <= 5.13 for r in pn)
for r in csv_rows("v1.6/perf-kq-n1.csv"):
    f = float(r["fast_med"]) / float(r["mmvq_med"]); q = float(r["mmq_med"]) / float(r["mmvq_med"])
    assert (.77 <= f <= 1.08 and 3.18 <= q <= 5.84) if r["type"] == "q4_K" else (.90 <= f <= 1.23 and 2.53 <= q <= 4.18), r
def tg(name):
    b = {}
    for r in csv_rows(name):
        b.setdefault((r["label"].rsplit("-r", 1)[0], int(r["depth"])), []).append(int(r["tg_tokens"]) * 1000 / float(r["tg_ms"]))
    assert all(len(v) == 5 for v in b.values()); return {k: median(v) for k, v in b.items()}
t2 = tg("v1.6/bench-2b-decode.csv")
for d in (-256, 256, 1024, 4096):
    assert 1.09 <= t2[("bi-v3", d)] / t2[("def", d)] <= 1.14 and 69.9 <= t2[("bi-v3", d)] <= 72.9
    assert .31 <= t2[("bi-mmq", d)] / t2[("def", d)] <= .33
t08 = tg("v1.6/bench-08b-decode.csv")
for d in (-256, 256, 1024, 4096):
    assert 1.03 <= t08[("bi-fast", d)] / t08[("def", d)] <= 1.071 and .34 <= t08[("bi-mmq", d)] / t08[("def", d)] <= .38
vb = {}
for r in csv_rows("v1.6/verify-batches-2b.csv"):
    if int(r["n"]) > 0: vb.setdefault((r["label"].rsplit("-r", 1)[0], int(r["n"])), []).append(float(r["ms_per_batch"]))
assert all(len(v) == 5 for v in vb.values())
vr = lambda c, n: median(vb[(c, n)]) / median(vb[("def", n)])
assert all(.84 <= vr("bi-nb", n) <= .93 and 1.66 <= vr("bi-mmq", n) <= 2.97 for n in (2, 3, 4, 5, 6, 8))
assert all(1.19 <= vr("bi-nb", n) <= 1.21 for n in (12, 16))
checks = (root / "v1.6/model-checks.txt").read_text()
assert checks.count("differ=0") == 5 and "DIFFERENT" not in checks and checks.count("112/112 tests passed") == 2
assert checks.count(": identical") == 24 and checks.count("not bit-identical to own ref: 0 of 34") == 4
assert {l.split()[0] for l in (root / "v1.6/dumps-q8v3-SHA256SUMS.txt").read_text().splitlines()} == {"ec330c4e6eae9d849c48330e2dd603817af68bb28096d13eae31f4d316e929cc"}
for name in ("e1-flag-on", "e1-flag-off"):
    assert len(csv_rows(f"v1.6/{name}.csv")) == 34
assert all(r["bit_identical_to_ref"] == "1" for r in csv_rows("v1.6/e1-flag-on.csv"))
for s in ("whole", "c4", "c8", "pair4"):
    lc = csv_rows(f"v1.6/logit-cmp/2b-singles-vs-{s}.csv")
    assert len(lc) == 1440 and all(float(r["max_abs_diff"]) == 0 for r in lc)
def tok(name):
    d = {}
    for line in (root / f"v1.6/spec/{name}.tok").read_text().splitlines():
        a = line.split(); d[(int(a[0]), int(a[1]))] = [x.split(":") for x in a[2:]]
    assert all(len(v) == 64 for v in d.values()); return d
def differ(d, k, i): return sum(any(x[i] != y[i] for x, y in zip(d[(0, s)], d[(k, s)])) for s in range(19))
for name in ("r1-bi-rs", "r1-bi-ckpt", "d1000-bi-rs"):   # invariant mode: speculation == plain, tokens and full-logit hashes
    d = tok(name); assert all(differ(d, k, 0) == differ(d, k, 1) == 0 for k, _ in d if k), name
d = tok("r1-def-rs")
assert differ(d, 1, 1) == 0 and all(differ(d, k, 1) == 19 and differ(d, k, 0) == 4 for k in (2, 3, 4, 5, 7))
d = tok("r1-def-ckpt"); assert differ(d, 1, 1) == 0 and all(differ(d, k, 1) == 19 and 4 <= differ(d, k, 0) <= 7 for k in (2, 3, 4, 5, 7))
d = tok("d1000-def-rs"); assert all(differ(d, k, 1) == 19 and differ(d, k, 0) == 0 for k in (3, 7))
pb, pd = tok("r1-bi-rs"), tok("r1-def-rs"); assert sum(any(x[0] != y[0] for x, y in zip(pb[(0, s)], pd[(0, s)])) for s in range(19)) == 3
for a in ("bi", "def"):   # plain decoding with n_rs_seq 8 (rs) and 0 (ckpt): identical token and logit-hash streams
    x, y = tok(f"r1-{a}-rs"), tok(f"r1-{a}-ckpt"); assert all(x[(0, s)] == y[(0, s)] for s in range(19)), a
ts = {}
for l in (root / "v1.6/spec/tok-SHA256SUMS.txt").read_text().splitlines():
    h, f = l.split(); ts.setdefault(f.split("-", 1)[1] if f.startswith("r") else f, set()).add(h)
assert all(len(ts[a + ".tok"]) == 1 for a in ("bi-rs", "def-rs", "bi-ckpt"))   # rounds 2 and 3 byte-identical to round 1
sp = {}
for f in (root / "v1.6/spec").glob("*.csv"):
    for r in csv_rows(f"v1.6/spec/{f.name}"):
        arm = r["label"].split("-", 1)[1] if r["label"].startswith("r") else r["label"]
        x = sp.setdefault((arm, int(r["k"]), r["label"]), [0, 0]); x[0] += int(r["n_gen"]); x[1] += float(r["ms"])
        assert r["redecode_bad"] == "0" or arm == "def-ckpt"
tps = lambda a, k: median(v[0] * 1000 / v[1] for (aa, kk, _), v in sp.items() if (aa, kk) == (a, k))
assert round(tps("bi-rs", 0), 1) == 66.5 and round(tps("def-rs", 0), 1) == 61.1 and round(tps("bi-ckpt", 0), 1) == 68.8
assert all(tps("bi-rs", k) < tps("bi-rs", 0) and tps("def-rs", k) < tps("bi-rs", 0) for k in (1, 2, 3, 4, 5, 7))   # plain invariant fastest
assert round(tps("d1000-bi-rs", 7) / tps("d1000-bi-rs", 0), 2) == 1.11

# v1.7.0: Q4_K/Q6_K N = 2..8 kernels, sanitizer, layout A/B
for name in ("v1.7/kq-nb-sweep-intree-float4.csv", "v1.7/kq-nb-sweep-intree-hybrid.csv"):   # every column of N = 1..8 vs N = 16 forced MMQ
    sw = [r for r in sweep_rows(name) if r["label"] == "sweep"]
    assert len(sw) == 450 and sum(int(r["bi_bad"]) for r in sw) == 0 and sum(int(r["bi_cmp"]) for r in sw) == 46812672, name
    assert sum(int(r["ctrl_default_bad"]) > 0 for r in sw) == 315 and sum(int(r["ref_negzero"]) for r in sw) == 12980
    assert all(int(r["ctrl_default_bad"]) > 0 for r in sw if r["family"] in ("normal", "zeros", "cancel", "single"))
gate = sweep_rows("v1.7/layout-ab-gate.csv")
assert {r["label"] for r in gate} == {"f4", "compact", "hybrid"} and sum(int(r["bi_bad"]) for r in gate) == 0 and sum(int(r["bi_cmp"]) for r in gate) == 2654208
for name in ("v1.7/model-checks-float4.txt", "v1.7/model-checks-hybrid.txt"):
    t = (root / name).read_text()
    assert "DIFFERENT" not in t and t.count("112/112 tests passed") == 2 and t.count("differ=0") == 2 and t.count(": identical") == 10, name
    assert "bad 0 of 46812672" in t and t.count("not bit-identical to own ref: 0 of 34") == 2
def pq(name):
    return [r for r in csv_rows(name) if int(r["m"]) >= 512]
rat = lambda r, a: float(r[a + "_med"]) / float(r["default_med"])
v1 = pq("v1.7/perf-nbkq-v1-float4.csv"); v2 = pq("v1.7/perf-nbkq-v2-compact.csv"); v3 = pq("v1.7/perf-nbkq-v3-hybrid.csv")
sel = lambda p, t, lo, hi: [r for r in p if r["type"] == t and lo <= int(r["n"]) <= hi]
assert max(rat(r, "bi-new") for r in sel(v1, "q4_K", 8, 8)) > 1.6                               # float4: the N = 8 slowdown
assert all(.72 <= rat(r, "bi-new") <= .978 for r in sel(v3, "q4_K", 8, 8)) and len(sel(v3, "q4_K", 8, 8)) == 12
assert all(.729 <= rat(r, "bi-new") <= 1.158 for r in sel(v3, "q4_K", 2, 2)) and len(sel(v3, "q4_K", 2, 2)) == 12
assert all(rat(r, "bi-new") < rat(r, "bi-mmq") for r in v3)                                     # hybrid always beats forced MMQ
assert all(.638 <= rat(r, "bi-new") <= 1.179 for r in sel(v2, "q4_K", 3, 6))
assert all(.657 <= rat(r, "bi-new") <= 1.32 for r in sel(v1, "q6_K", 2, 8))
ptx = (root / "v1.7/ptxas-nb-kq.txt").read_text()
assert "q4_K NC=8 CP=0\tUsed 138 registers, 38400 bytes smem" in ptx and "q4_K NC=8 CP=1\tUsed 150 registers, 19392 bytes smem" in ptx
assert ptx.count("0 bytes spill stores, 0 bytes spill loads") == 21
def spec7(tag):
    s = {}
    for f in (root / f"v1.7/spec/{tag}").glob("r*-bi-*.csv"):
        for r in csv_rows(f"v1.7/spec/{tag}/{f.name}"):
            x = s.setdefault((r["label"].split("-", 1)[1], int(r["k"])), {}).setdefault(r["label"], [0, 0])
            x[0] += int(r["n_gen"]); x[1] += float(r["ms"])
            assert r["redecode_bad"] == "0"
    assert all(len(v) == 3 for v in s.values()), tag
    return {k: median(n * 1000 / ms for n, ms in v.values()) for k, v in s.items()}
s1, s3, sab = spec7("v1-float4"), spec7("v3-hybrid"), spec7("layout-ab")
assert round(s1[("bi-new", 1)], 1) == 70.0 and round(s1[("bi-off", 1)], 1) == 52.8 and round(s3[("bi-new", 1)], 1) == 68.9
assert all(1.02 <= s[(a, 1)] / s[(a, 0)] <= 1.06 for s, a in ((s1, "bi-new"), (s3, "bi-new"), (sab, "bi-hybrid"), (sab, "bi-f4"), (sab, "bi-compact")))
assert max(sab[(a, 3)] for a in ("bi-hybrid", "bi-f4", "bi-compact")) / min(sab[(a, 3)] for a in ("bi-hybrid", "bi-f4", "bi-compact")) < 1.02   # layout effect within noise
ref = tok("r1-bi-rs")   # v1.6.0 plain invariant decoding
for tag, names in (("v1-float4", ("r1-bi-new", "r1-bi-off")), ("v3-hybrid", ("r1-bi-new", "r1-bi-off")), ("layout-ab", ("r1-bi-hybrid", "r1-bi-f4", "r1-bi-compact"))):
    for n in names:
        d = {}
        for line in (root / f"v1.7/spec/{tag}/{n}.tok").read_text().splitlines():
            a = line.split(); d[(int(a[0]), int(a[1]))] = [x.split(":") for x in a[2:]]
        assert all(d[(k, s)] == ref[(0, s)] for (k, s) in d), (tag, n)   # every k: tokens + full-logit hashes == plain
san = (root / "v1.7/sanitizer/summary.txt").read_text().splitlines()
assert len(san) == 20 and all("0 errors" in l or "0 hazards" in l for l in san if "initcheck" not in l)
for n in ("q4K-1024x100", "q6K-3584x100", "q8-2048x100"):   # all initcheck reports in upstream's forced-MMQ kernel, none in the new kernels
    dg = (root / f"v1.7/sanitizer/{n}-initcheck-full.digest.txt").read_text()
    assert "224 ========= Uninitialized __global__ memory read" in dg and "bi_n" not in dg and "mul_mat_q<" in dg, n
assert "224 ========= Uninitialized __global__ memory read" in (root / "v1.7/sanitizer/prhead-q4K-initcheck.digest.txt").read_text()
print("Archived headline aggregates, CPU oracle comparisons, measured sweep points, rerun, v1.1.0, v1.2.0, v1.3.0, v1.3.1, v1.3.2, v1.4.0, v1.5.0, v1.5.1, v1.5.2, v1.6.0 and v1.7.0 controls verified.")
