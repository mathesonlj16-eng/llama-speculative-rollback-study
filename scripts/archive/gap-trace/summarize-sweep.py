"""Summarizes a perturbation sweep CSV by site, layer and perturbation size."""
import csv
import statistics as st
import sys
from collections import defaultdict

g = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    g[(r["key"], int(r["index"]), int(r["layer"]), float(r["size"]))].append(r)
print("controls bit-identical:", all(r["bit_equal"] == "1" for k, v in g.items() if k[3] == 0 for r in v),
      "| injections applied:", all(r["hits"] == "1" for k, v in g.items() if k[3] > 0 for r in v))
print("key,index,layer,size,median_output_rms,gain,median_max_diff,max_max_diff,top_flips,distinct_outputs")
for (key, idx, layer, size), rows in sorted(g.items()):
    if size == 0:
        continue
    a = st.median(float(r["input_rms"]) for r in rows)
    b = st.median(float(r["rms_diff"]) for r in rows)
    print(f"{key},{idx},{layer},{size:g},{b:.3g},{b / a if a else 0:.3g},{st.median(float(r['max_abs_diff']) for r in rows):.4f},"
          f"{max(float(r['max_abs_diff']) for r in rows):.4f},{sum(r['top'] != r['base_top'] for r in rows)},{len({r['hash'] for r in rows})}")
