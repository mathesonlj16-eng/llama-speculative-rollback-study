"""v1.4.0: per-sequence summary of logit-cmp CSVs. Input: fixture TSV, concatenated CSV with a leading phase column, output path."""
import csv, sys
tsv, raw, out = sys.argv[1:4]
L = {}
for line in open(tsv, encoding="utf-8"):
    k, p, g = line.rstrip("\n").split("\t"); L[k] = (len(p.split()), len(g.split()))
groups = {}
for r in csv.DictReader(open(raw)):
    groups.setdefault((r["phase"], r["label"], r["key"]), []).append(r)
with open(out, "w", newline="\n") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["phase", "comparison", "key", "prompt_tokens", "gen_tokens", "positions", "nonzero", "first_nonzero_index", "max_abs_diff", "argmax_differs"])
    for (ph, lab, k), rs in groups.items():
        nz = [int(r["index"]) for r in rs if float(r["max_abs_diff"]) != 0]
        w.writerow([ph, lab, k, L[k][0], L[k][1], len(rs), len(nz), nz[0] if nz else "", "%.6g" % max(float(r["max_abs_diff"]) for r in rs),
                    sum(r["top_a"] != r["top_b"] for r in rs)])
