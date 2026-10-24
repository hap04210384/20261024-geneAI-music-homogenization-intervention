# -*- coding: utf-8 -*-
"""Aggregate gate batches into the final compliance report.

For each (family, condition, start) the LATEST timestamped batch wins
(parser/post-processing fixes invalidate earlier batches). Cells aggregate
across starts. Output: runs/c/gate_final.json + console table.
2026-10-23. Stage C of the seven-stage pipeline.
"""
import json, os, re, glob, collections

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "runs", "c")

PAT = re.compile(r"gate_(\w+)_(\w+)_(\d{8}-\d{6})(?:_s(\d+))?\.json")

def main():
    latest = {}
    for f in glob.glob(os.path.join(RUNS, "gate_*.json")):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        fam, cond, stamp, start = (m.group(1), m.group(2), m.group(3),
                                   int(m.group(4) or 0))
        key = (fam, cond, start)
        if key not in latest or stamp > latest[key][0]:
            latest[key] = (stamp, f)
    agg = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for (fam, cond, start), (stamp, f) in sorted(latest.items()):
        d = json.load(open(f, encoding="utf-8"))
        agg[(fam, cond)][0] += d["compliant"]
        agg[(fam, cond)][1] += d["n"]
        for k, v in d["failures"].items():
            agg[(fam, cond)][2][k] += v
    cells = []
    print(f"{'cell':<22}{'compliant':>10}{'n':>5}{'rate':>8}  failures")
    for (fam, cond), (ok, n, fails) in sorted(agg.items()):
        cells.append(dict(family=fam, condition=cond, compliant=ok, n=n,
                          rate=round(ok / n, 3), failures=dict(fails)))
        print(f"{fam+'/'+cond:<22}{ok:>10}{n:>5}{ok/n:>8.3f}  {dict(fails)}")
    out = os.path.join(RUNS, "gate_final.json")
    json.dump(dict(date="2026-10-23", gate="ABC parse compliance, prompt v1",
                   cells=cells), open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("->", out)

if __name__ == '__main__':
    main()
