# -*- coding: utf-8 -*-
"""
Stage A4 step 1: export window token-sets as FIMI transaction files.

One FIMI file per (corpus, alphabet). Item ids are 1-based dense integers
(id 0 is reserved: the engines mishandle it -- see adapter design notes);
the token<->id mapping is saved next to each file for reproducibility.

Semantics note (frozen): the engine's frequency is WINDOW-level (a pattern
scores once per window that contains it). The manuscript's document
frequency pi is PIECE-level (a pattern scores once per piece that contains
it in >=1 window); piece-level pi is recomputed in a post-pass
(compute_pi.py) after mining.

2026-10-22.
"""
import gzip, json, os, sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "data_backup")
DERIVED = os.path.join(DATA, "derived")
FIMI = os.path.join(DERIVED, "fimi")
os.makedirs(FIMI, exist_ok=True)

def export(corpus):
    src = os.path.join(DERIVED, f"{corpus}.windows.jsonl.gz")
    for alph in ("a1", "a2", "a3"):
        tok2id = {}
        n_tx = 0
        out = os.path.join(FIMI, f"{corpus}.{alph}.fimi")
        with open(out, "w", encoding="utf-8") as w:
            with gzip.open(src, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if not r.get("kept", True):
                        continue          # short (<8 events) or exact duplicate
                    for win in r[alph]:
                        ids = set()
                        for tok in win:
                            i = tok2id.get(tok)
                            if i is None:
                                i = len(tok2id) + 1      # 1-based dense
                                tok2id[tok] = i
                            ids.add(i)
                        w.write(" ".join(str(x) for x in sorted(ids)))
                        w.write("\n")
                        n_tx += 1
        mp = os.path.join(FIMI, f"{corpus}.{alph}.items.json")
        with open(mp, "w", encoding="utf-8") as w:
            json.dump({"alphabet": alph, "n_transactions": n_tx,
                       "token_of_id": {str(i): t for t, i in sorted(tok2id.items(), key=lambda kv: kv[1])}},
                      w, ensure_ascii=False, indent=1)
        print(f"[export] {corpus}.{alph}: {n_tx} transactions, {len(tok2id)} items")

if __name__ == "__main__":
    for c in (sys.argv[1:] or ("mtc", "pop909")):
        export(c)
