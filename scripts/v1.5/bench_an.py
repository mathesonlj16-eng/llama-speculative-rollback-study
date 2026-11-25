"""v1.5.0: medians and ratios from bench-bi timing rows (warm-up rows have negative depth and are skipped)."""
import csv, sys, statistics as st
rows = [r for r in csv.DictReader(open(sys.argv[1])) if int(r["depth"]) > 0]
cfgs = ["base-def", "base-bi", "fafix-bi", "fafix-def"]
med = {}; spread = {}
for cfg in cfgs:
    for d in (256, 1024, 4096):
        rs = [r for r in rows if r["label"].rsplit("-r", 1)[0] == cfg and int(r["depth"]) == d]
        pp = [d / float(r["pp_ms"]) * 1000 for r in rs]; tg = [64 / float(r["tg_ms"]) * 1000 for r in rs]
        med[cfg, d] = (st.median(pp), st.median(tg)); spread[cfg, d] = (len(rs), (max(tg) - min(tg)) / st.median(tg) * 100, (max(pp) - min(pp)) / st.median(pp) * 100)
print("cfg,depth,n,pp_tok_s,tg_tok_s,tg_range_pct,pp_range_pct")
for (cfg, d), (pp, tg) in med.items():
    n, tr, pr = spread[cfg, d]; print(f"{cfg},{d},{n},{pp:.1f},{tg:.2f},{tr:.1f},{pr:.1f}")
print("depth,pp fafix-bi/base-bi,tg fafix-bi/base-bi,pp base-bi/base-def,tg base-bi/base-def,pp fafix-def/base-def,tg fafix-def/base-def")
for d in (256, 1024, 4096):
    r = lambda a, b, i: med[a, d][i] / med[b, d][i]
    print(f"{d},{r('fafix-bi','base-bi',0):.3f},{r('fafix-bi','base-bi',1):.3f},{r('base-bi','base-def',0):.3f},{r('base-bi','base-def',1):.3f},{r('fafix-def','base-def',0):.3f},{r('fafix-def','base-def',1):.3f}")
