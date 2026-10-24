# -*- coding: utf-8 -*-
"""Stage F step 3: post-process seeded samples (parse -> encode -> windows).

Same code path as stage C post-processing; M1 (continuation) cells have the
seeded opening prepended before parsing. No manipulation check here (stage F
gates on the stage-C majority rule already established).

Output: runs/f/encoded/<cell>.windows.jsonl.gz   (cell = fam_mech_arm_cond_c)
2026-10-24.
"""
import gzip, json, os, sys, time, collections
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "encode"))
ROOT = Path(__file__).resolve().parents[3]
RUNF = ROOT / "runs" / "f"
ENC = RUNF / "encoded"
ENC.mkdir(parents=True, exist_ok=True)

from gate import extract_abc
from abc_parse import parse_abc, ABCError
import build_windows as bw


def load_tonal_map():
    return bw.learn_mtc_tonal_map()


def tonal_token(nt, tonic_pc, mode, tmap):
    return tmap.get((mode, (nt['pitch'] - tonic_pc) % 12))


def opening_abc(cond, c):
    """Seeded opening for M1 cells (own seeds of the prompted culture)."""
    d = json.loads((RUNF / f"seeds_{cond}.json").read_text(encoding="utf-8"))
    frags = d["cells"][str(c)]["abc_fragments"]
    return ("X:1\nT:Seeded Opening\nM:4/4\nL:1/4\nK:C\n" + " | ".join(frags) + " |")


def encode_cell(cell, tmap):
    src = RUNF / f"sample_{cell}.jsonl"
    dst = ENC / f"{cell}.windows.jsonl.gz"
    if dst.exists():
        print(f"[post_f] {cell} already encoded, skip", flush=True)
        return
    mech = cell.split("_")[1]
    cond = cell.split("_")[3]
    c = cell.split("_")[4]
    stats = collections.Counter()
    seen = set()
    rows = []
    for ln in open(src, encoding="utf-8"):
        rec = json.loads(ln)
        if rec.get("failed"):
            stats['api_failed'] += 1
            continue
        stats['raw'] += 1
        text = rec["text"]
        if mech == "M1":
            text = opening_abc(cond, c) + "\n" + text
        try:
            notes, segs, freemeter, meta = parse_abc(extract_abc(text))
            stats['parsed'] += 1
        except ABCError as e:
            stats['parse_fail:' + str(e)[:40]] += 1
            continue
        if len(notes) < 8:
            stats['short'] += 1
            continue
        sig = bw.sig_of(notes)
        if sig in seen:
            stats['dup'] += 1
            continue
        seen.add(sig)
        tonic_pc, mode = meta['key'], meta['mode']
        for nt in notes:
            nt['token'] = tonal_token(nt, tonic_pc, mode, tmap)
        upto = max(nt['onset'] for nt in notes)
        grid = bw.bar_grid(upto, segs)
        wins = bw.split_windows([nt['onset'] for nt in notes], grid)
        a1, a2, a3 = [], [], []
        for idx in wins:
            x1, x2, x3 = bw.code_window(idx, notes)
            a1.append(x1); a2.append(x2); a3.append(x3)
        rows.append(dict(id=f"{cell}_{rec['i']}", model=rec['model'],
                         cell=cell, mech=mech, arm=rec['arm'], condition=cond,
                         dose=float(c), i=rec['i'],
                         key=tonic_pc, mode=mode,
                         n_events=len(notes), n_windows=len(wins),
                         a1=a1, a2=a2, a3=a3))
        stats['kept'] += 1
        stats['windows'] += len(wins)
    with gzip.open(dst, 'wt', encoding='utf-8') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[post_f] {cell}: {dict(stats)} -> {dst.name}", flush=True)


def main():
    tmap = load_tonal_map()
    cells = sorted(p.stem.replace("sample_", "")
                   for p in RUNF.glob("sample_*.jsonl"))
    print(f"[post_f] {len(cells)} cells with sample files", flush=True)
    for cell in cells:
        encode_cell(cell, tmap)
    print("[post_f] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
