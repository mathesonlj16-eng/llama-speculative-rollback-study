import collections
from bisect import bisect_right
import json
from pathlib import Path
import sqlite3
import sys

def union_ms(events):
    total = end = 0
    for a, b, *_ in sorted(events):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return round(total / 1e6, 3)

for filename in sys.argv[1:]:
    path = Path(filename).resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    req = json.loads(path.with_name(path.stem + "-requests.json").read_text())
    def rows(sql):
        return list(db.execute(sql))
    print(path)
    print("request elapsed and event offsets", (req["after_ns"]-req["before_ns"])/1e6, [(x["monotonic_ns"]-req["before_ns"])/1e6 for x in req["events"][:4]])
    print("control", req["control"]["timings"], "trace", req["events"][-1]["event"].get("timings"))
    api = rows("select start,end,value,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME join StringIds on id=nameId order by start")
    copies = rows("select start,end,copyKind,bytes,correlationId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY order by start")
    kern = rows("select start,end,value,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL join StringIds on id=shortName order by start")
    logits = [c for c in copies if c[2] == 2 and c[3] % (248320*4) == 0]
    print("first copies >= 500000 bytes", [c for c in copies if c[3] >= 500000][:16])
    print("first/last logits", logits[:10], logits[-5:])
    print("first/last api", api[:8], api[-6:])
    if not logits:
        continue
    lo, hi = logits[0][1], max(x[1] for group in (api, copies, kern) for x in group)
    print("window ns", lo, hi, "ms", (hi-lo)/1e6)
    api, copies, kern = ([x for x in group if lo <= x[0] < hi] for group in (api, copies, kern))
    def summarize(events, limit=20, byte_column=None):
        groups = collections.defaultdict(list)
        for x in events:
            groups[x[2]].append(x)
        durations = [(key, len(g), sum(b-a for a,b,*_ in g)/1e6, sum(x[byte_column] for x in g) if byte_column is not None else None) for key,g in groups.items()]
        return sorted(durations, key=lambda x:-x[2])[:limit]
    print("API", summarize(api))
    print("copy", summarize(copies, byte_column=3))
    print("kernel", len(kern), sum(b-a for a,b,*_ in kern)/1e6, summarize(kern, 10))
    print("device union ms", union_ms(kern+copies))
    if not rows("select name from sqlite_master where name='NVTX_EVENTS'"):
        continue
    ranges = rows("select start,end,coalesce(text,value) from NVTX_EVENTS left join StringIds on textId=id where end is not null order by start")
    print("first/last ranges", ranges[:12], ranges[-8:])
    ctx_target = next(x[2].split()[1] for x in ranges if x[2].startswith("process") and x[0] < lo)
    ranges = [x for x in ranges if lo <= x[0] < hi]
    pending = False
    phases = collections.defaultdict(list)
    for a,b,label in ranges:
        phase, ctx, flags, size = label.split()
        model = "target" if ctx == ctx_target else "draft"
        if phase == "state_restore" and model == "target" and flags == "1":
            pending = True
        if phase == "process" and model == "target" and pending:
            phase = "replay"
            pending = False
        phases[model,phase,flags,size].append((a,b))
    api_by_id = {x[3]:x for x in api}
    checkpoints = [(a,b) for a,b,label in ranges if label.startswith(("state_save ", "state_restore "))]
    clipped = [(max(a,x[0]),min(b,x[1])) for a,b in checkpoints for x in kern if x[0] < b and x[1] > a]
    print("Checkpoint API wall ms", union_ms(checkpoints), "overlapping GPU kernel execution ms", union_ms(clipped))
    for key, intervals in phases.items():
        starts = [a for a,b in intervals]
        def within(t):
            i = bisect_right(starts, t) - 1
            return i >= 0 and t < intervals[i][1]
        phase_api = [x for x in api if within(x[0])]
        phase_copies = [x for x in copies if x[4] in api_by_id and within(api_by_id[x[4]][0])]
        phase_kern = [x for x in kern if x[3] in api_by_id and within(api_by_id[x[3]][0])]
        completion = collections.defaultdict(int)
        for x in phase_copies + phase_kern:
            correlation = x[4] if isinstance(x[2], int) else x[3]
            i = bisect_right(starts, api_by_id[correlation][0]) - 1
            completion[i] = max(completion[i], x[1])
        extended = [(a,max(b,completion[i])) for i,(a,b) in enumerate(intervals)]
        print(key, len(intervals), union_ms(intervals), "through GPU completion union ms", union_ms(extended), "copies", summarize(phase_copies, byte_column=3),
              "kernel", len(phase_kern), sum(b-a for a,b,*_ in phase_kern)/1e6, "api", summarize(phase_api, 6))
