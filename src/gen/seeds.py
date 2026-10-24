# -*- coding: utf-8 -*-
"""
Stage F step 1: build and FREEZE the seed sets (protocol 6a, amendment F-1).

Per culture and nominal coverage c in {0.1, 0.25, 0.5, 1.0}:
  greedy set cover of that culture's kept pieces using the culture's own
  a3-short23 pattern inventory (census sizes 2..3), ranked by marginal
  uncovered-piece coverage; stop when the covered piece fraction >= c.
  Listing cap 250 fragments (coverage-priority truncation; actual coverage
  reported as c_act).

Rendering (canonical form -- patterns are unordered sets; this fixed order is
part of the frozen protocol and its limitation is acknowledged in 6a):
  root C4; each token "iv|dur" adds signed-semitone iv to the running pitch;
  tokens ordered by (iv, dur); note spelled with sharps; durations
  Whole=4 Half=2 Quarter=(none) Eighth=/2 Sixteenth=/4 Other=1 (L:1/4).

Output: runs/f/seeds_<culture>.json  {culture, c_nominal, seeds[[tokens]],
        abc_fragments[str], n_seeds, c_act, n_pieces_target}
        runs/f/seeds_manifest.json     protocol + code md5 + build stats
2026-10-24.
"""
import gzip, json, os, sys, hashlib, time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = Path(__file__).resolve().parents[3]      # …/Manuscript_work
DERIVED = ROOT / "data_backup" / "derived"
RUNF = ROOT / "runs" / "f"
RUNF.mkdir(parents=True, exist_ok=True)

CULTURES = {"CN": "pop909", "EU": "mtc"}
N_PIECES = {"CN": 908, "EU": 17638}
DOSES = (0.1, 0.25, 0.5, 1.0)
LIST_CAP = 250

DUR_ABC = {"Whole": "4", "Half": "2", "Quarter": "", "Eighth": "/2",
           "Sixteenth": "/4", "Other": "1"}
PC_SPELL = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]


def a3_short_patterns(corpus):
    """The culture's a3 census patterns of size 2..3 as token tuples."""
    with gzip.open(DERIVED / f"reflib.{corpus}.a3.json.gz", "rt", encoding="utf-8") as f:
        rec = json.load(f)
    items = rec["items"]
    pats = set()
    for key in rec["census"]:
        ids = key.split("|")
        if 2 <= len(ids) <= 3:
            pats.add(tuple(sorted(items[i] for i in ids)))
    return sorted(pats)


def load_windows_a3(corpus):
    pieces = []
    with gzip.open(DERIVED / f"{corpus}.windows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("kept", True):
                continue
            pieces.append([frozenset(w) for w in r["a3"]])
    return pieces


def render_fragment(tokens):
    """Canonical ABC rendering of one pattern. The pattern is an unordered
    set of (signed-semitone, duration) steps; the frozen canonical form
    (amendment F-2, adopted before any sampling) spells the fragment in the
    transposition that minimises accidentals (ties broken by lowest
    pitches); tokens ordered by (iv, dur)."""
    steps = sorted((int(t.split("|")[0]), t.split("|")[1]) for t in tokens)
    best = None
    for shift in range(12):
        pcs = []
        pc = shift
        for iv, _ in steps:
            pc += iv
            pcs.append(pc)
        acc = sum(1 for p in pcs if len(PC_SPELL[p % 12]) > 1)
        hi = max(pcs) if pcs else 0
        key = (acc, hi)
        if best is None or key < best[0]:
            best = (key, pcs)
    pcs = best[1]
    out = []
    for (iv, dur), pc in zip(steps, pcs):
        midi = 60 + pc
        letter = PC_SPELL[midi % 12]
        base = letter[0]
        acc_m = letter[1:] if len(letter) > 1 else ""
        name = (base.upper() if midi < 72 else base.lower()) + acc_m
        out.append(name + DUR_ABC.get(dur, "1"))
    return " ".join(out)


def greedy_cover(corpus):
    pieces = load_windows_a3(corpus)
    pats = a3_short_patterns(corpus)
    n = len(pieces)
    cache = RUNF / f"hcache_seeds_{corpus}.npy"
    if cache.exists():
        H = np.load(cache)
    else:
        import stage_d
        t = time.time()
        H = stage_d.build_hits(pieces, pats)
        print(f"[seeds] hit matrix {corpus}: {H.shape} in {time.time()-t:.0f}s",
              flush=True)
        np.save(cache, H)
    del pieces
    Hs = H.T.astype(np.int8)
    del H
    uncovered = np.ones(n, dtype=np.int8)
    import heapq
    gains = Hs.dot(uncovered)                     # one full pass
    heap = [(-float(g), j) for j, g in enumerate(gains) if g > 0]
    heapq.heapify(heap)
    order, cum = [], []
    stale_marker = {}
    while heap and len(order) < LIST_CAP:
        neg, j = heapq.heappop(heap)
        g_new = int(Hs[j].dot(uncovered))
        if g_new <= 0:
            continue
        if -neg > g_new:                          # stale upper bound
            heapq.heappush(heap, (-float(g_new), j))
            continue
        order.append(j)
        uncovered &= (1 - Hs[j]).astype(np.int8)
        cum.append(1.0 - float(uncovered.mean()))
    return pats, order, cum, n


def main():
    t0 = time.time()
    manifest = {"protocol": "6a/F-1", "date": "2026-10-24", "doses": list(DOSES),
                "list_cap": LIST_CAP, "cultures": CULTURES, "cells": {}}
    for culture, corpus in CULTURES.items():
        out = RUNF / f"seeds_{culture}.json"
        if out.exists():
            print(f"[seeds] {culture} already frozen, skip")
            continue
        pats, order, cum, n = greedy_cover(corpus)
        cells = {}
        for c in DOSES:
            k = next((i for i, f in enumerate(cum) if f >= c), len(order) - 1)
            k = min(k + 1, len(order))            # 1-based prefix length
            k = min(k, LIST_CAP)
            chosen = order[:k]
            cov = cum[k - 1]
            toks = [list(pats[j]) for j in chosen]
            frags = [render_fragment(pats[j]) for j in chosen]
            cells[str(c)] = {"seeds": toks, "abc_fragments": frags,
                             "n_seeds": k, "c_act": round(float(cov), 4),
                             "n_pieces_target": n}
            print(f"[seeds] {culture} c={c}: {k} seeds, actual coverage "
                  f"{cov:.3f} (target {c})", flush=True)
        manifest["cells"][culture] = cells
        (RUNF / f"seeds_{culture}.json").write_text(
            json.dumps({"culture": culture, "cells": cells}, ensure_ascii=False),
            encoding="utf-8")
    here = Path(__file__)
    manifest["code_md5"] = hashlib.md5(here.read_bytes()).hexdigest()
    manifest["elapsed_sec"] = round(time.time() - t0, 1)
    (RUNF / "seeds_manifest.json").write_text(json.dumps(manifest, indent=1),
                                              encoding="utf-8")
    print("[seeds] manifest ->", RUNF / "seeds_manifest.json")


if __name__ == "__main__":
    main()
