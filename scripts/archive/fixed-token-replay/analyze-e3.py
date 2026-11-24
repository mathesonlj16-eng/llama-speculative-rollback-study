"""Re-rank E3 using only tensors whose captured final-token slice is valid.

Excluded: the inactive mixed-input branch (nodes 1-3), carried recurrent state, and view-like
ops, because E3 reads tensor bytes as contiguous and those ops can have permuted strides.
"""
import csv
import re
import sys

SKIP_OPS = {"VIEW", "RESHAPE", "TRANSPOSE", "PERMUTE"}
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
valid = []
for r in rows:
    if r["comparable"] != "1" or r["op"] in SKIP_OPS or int(r["order"]) in (1, 2, 3):
        continue
    if re.search(r"cache_|conv_states|state_predelta", r["tensor"]):
        continue
    if not re.search(r"axis[0-3]$", r["shape_7+4"]):
        continue
    r["d"] = float(r["max_abs_diff"])
    r["m"] = float(r["max_abs_value"])
    r["rel"] = r["d"] / r["m"] if r["m"] else 0.0
    valid.append(r)

print(f"valid tensors: {len(valid)} of {len(rows)}")
print("\nfirst 30 valid tensors in execution order:")
for r in valid[:30]:
    print(f"{r['order']:>5} {r['tensor'][:34]:34} {r['op']:12} {r['shape_7+4']:24} d={r['d']:.4g} rel={r['rel']:.3g}")
print("\nfirst tensors exceeding relative thresholds:")
for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    hit = next((r for r in valid if r["rel"] > t), None)
    print(f"  rel>{t:g}:", f"{hit['order']} {hit['tensor']} {hit['op']} d={hit['d']:.4g} rel={hit['rel']:.3g}" if hit else "none")
