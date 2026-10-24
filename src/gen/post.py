# -*- coding: utf-8 -*-
"""Stage C post-processing: parse -> encode -> manipulation check.

Pipeline (runs after sampling, resumable per cell):
  1. parse    raw JSONL -> event streams via abc_parse + gate.extract_abc
              (frozen dialect normalisation); failures counted as compliance
              dropouts and reported per cell.
  2. dedup    exact event-stream signature, within cell (same conservative
              rule as the human corpora); <8-event pieces dropped (A2 rule).
  3. encode   windowing + three alphabets, identical code path as Stage A
              (imports build_windows); A1 tonal tokens via the MTC-learned
              convention map, same as POP909.
  4. manip    per-piece majority vote with the frozen manip_clf_{a1,a2,a3}
              models; one-sided binomial per cell, BH across cells.

Output: runs/c/encoded/<fam>_<cond>.windows.jsonl.gz
        runs/c/manip_results.json + printed report
MD5 manifest: runs/c/manifest_post.md5

2026-10-23. Stage C of the seven-stage pipeline.
"""
import gzip, json, os, sys, time, hashlib, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "runs", "c")
ENC = os.path.join(RUNS, "encoded")
os.makedirs(ENC, exist_ok=True)

from gate import extract_abc
from abc_parse import parse_abc, ABCError
sys.path.insert(0, os.path.join(HERE, "..", "encode"))
import build_windows as bw

FAMILIES = ("deepseek", "dashscope", "zhipu")
CONDS = ("CN", "EU")


def load_tonal_map():
    return bw.learn_mtc_tonal_map()


def tonal_token(nt, tonic_pc, mode, tmap):
    t = tmap.get((mode, (nt['pitch'] - tonic_pc) % 12))
    return t


def encode_cell(fam, cond, tmap):
    src = os.path.join(RUNS, f"sample_{fam}_{cond}.jsonl")
    dst = os.path.join(ENC, f"{fam}_{cond}.windows.jsonl.gz")
    if os.path.exists(dst):
        print(f"[post] {fam}/{cond} already encoded, skip")
        return
    stats = collections.Counter()
    seen = set()
    rows = []
    for ln in open(src, encoding="utf-8"):
        rec = json.loads(ln)
        if rec.get("failed"):
            stats['api_failed'] += 1
            continue
        stats['raw'] += 1
        try:
            notes, segs, freemeter, meta = parse_abc(extract_abc(rec["text"]))
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
        rows.append(dict(id=f"{fam}_{cond}_{rec['i']}", model=rec['model'],
                         family=fam, condition=cond, i=rec['i'],
                         key=tonic_pc, mode=mode,
                         n_events=len(notes), n_windows=len(wins),
                         a1=a1, a2=a2, a3=a3))
        stats['kept'] += 1
        stats['windows'] += len(wins)
    with gzip.open(dst, 'wt', encoding='utf-8') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[post] {fam}/{cond}: {dict(stats)} -> {os.path.basename(dst)}")


class FrozenMNB:
    def __init__(self, ab):
        d = np.load(os.path.join(RUNS, f"manip_clf_{ab}.npz"), allow_pickle=True)
        self.ll, self.lp = d['ll'], d['lp']
        self.vocab = {t: i for i, t in enumerate(d['vocab'].tolist())}

    def predict_windows(self, windows):
        X = np.zeros((len(windows), len(self.vocab)), dtype=np.int32)
        for r, toks in enumerate(windows):
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    X[r, j] += 1
        s = X @ self.ll.T + self.lp
        return (s[:, 1] > s[:, 0]).astype(int)   # 1 = CN


def binom_sf(k, n, p=0.5):
    """P[X >= k] for X~Bin(n,p), exact."""
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def manip_check():
    results = []
    for fam in FAMILIES:
        clfs = {ab: FrozenMNB(ab) for ab in ("a1", "a2", "a3")}
        for cond in CONDS:
            path = os.path.join(ENC, f"{fam}_{cond}.windows.jsonl.gz")
            if not os.path.exists(path):
                continue
            cell = dict(family=fam, condition=cond, alphabets={})
            for ab in ("a1", "a2", "a3"):
                votes = {}
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    for ln in f:
                        rec = json.loads(ln)
                        pred = clfs[ab].predict_windows(rec[ab])
                        if len(pred) == 0:
                            continue
                        cn = int((pred == 1).sum())
                        votes[rec['id']] = cn * 2 > len(pred)   # majority CN
                n = len(votes)
                k_cn = sum(votes.values())
                # for CN cells the prediction should BE CN; for EU cells the
                # informative quantity is k_eu = n - k_cn
                k = k_cn if cond == "CN" else n - k_cn
                p = binom_sf(k, n)
                cell['alphabets'][ab] = dict(n_pieces=n,
                                              frac_culture=k / max(n, 1),
                                              p_one_sided=round(p, 6))
            results.append(cell)
    # BH across all (family, condition, alphabet) tests
    tests = []
    for c in results:
        for ab, r in c['alphabets'].items():
            tests.append((c['family'], c['condition'], ab, r['p_one_sided']))
    m = len(tests)
    order = sorted(tests, key=lambda t: t[3])
    q = {}
    prev = 1.0
    for rank, (fam, cond, ab, p) in reversed(list(enumerate(order, 1))):
        val = min(prev, p * m / rank)
        q[(fam, cond, ab)] = round(val, 6)
        prev = val
    for c in results:
        for ab in c['alphabets']:
            c['alphabets'][ab]['q_BH'] = q[(c['family'], c['condition'], ab)]
    out = os.path.join(RUNS, "manip_results.json")
    json.dump(dict(date="2026-10-23", protocol="v2",
                   rule="one-sided binomial vs 0.5 at piece level; "
                        "BH across all tests",
                   cells=results),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[post] manipulation results -> {out}")
    for c in results:
        for ab, r in c['alphabets'].items():
            print(f"  {c['family']}/{c['condition']}/{ab}: "
                  f"n={r['n_pieces']} frac={r['frac_culture']:.3f} "
                  f"p={r['p_one_sided']} q={r['q_BH']}")


def write_manifest():
    man = os.path.join(RUNS, "manifest_post.md5")
    with open(man, "w", encoding="utf-8") as w:
        for fn in sorted(os.listdir(ENC)):
            p = os.path.join(ENC, fn)
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            w.write(f"{h}  {fn}\n")
        for fn in ("post.py", "protocol_v2.json", "manip_results.json"):
            p = os.path.join(RUNS, fn) if fn != "post.py" else os.path.join(
                HERE, "post.py")
            if os.path.exists(p):
                w.write(f"{hashlib.md5(open(p,'rb').read()).hexdigest()}  {fn}\n")
    print("[post] manifest ->", man)


def main():
    tmap = load_tonal_map()
    print(f"[post] tonal map: {len(tmap)} conventions")
    for fam in FAMILIES:
        for cond in CONDS:
            src = os.path.join(RUNS, f"sample_{fam}_{cond}.jsonl")
            if os.path.exists(src):
                encode_cell(fam, cond, tmap)
    manip_check()
    write_manifest()
    print("[post] DONE")


if __name__ == '__main__':
    main()
