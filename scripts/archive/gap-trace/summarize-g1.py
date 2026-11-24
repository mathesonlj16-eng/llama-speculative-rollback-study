"""Summarizes G1 by backend, cache type, flash attention, sequence and batch shape."""
import csv
import glob
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

groups = defaultdict(list)
for p in sorted(glob.glob(str(Path(sys.argv[1]) / "*.csv"))):
    for r in csv.DictReader(open(p)):
        groups[(r["config"], r["key"], r["shape"])].append(r)
print("config,key,shape,positions,max_abs_diff,median_max_abs_diff,top1_changes_vs_singles,largest_gen_gap,gap_at_11,gap_at_56")
for (config, key, shape), rows in groups.items():
    if shape == "singles":
        continue
    d = [float(r["max_abs_diff_vs_singles"]) for r in rows]
    changes = sum(r["top"] != r["ref_top"] for r in rows)
    print(f"{config},{key},{shape},{len(rows)},{max(d):.4f},{st.median(d):.4f},{changes},{max(float(r['gen_gap']) for r in rows):.4f},"
          f"{float(rows[11]['gen_gap']):.4f},{float(rows[56]['gen_gap']):.4f}")
