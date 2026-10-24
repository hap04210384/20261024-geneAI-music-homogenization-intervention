# -*- coding: utf-8 -*-
"""Manipulation-check classifier (stage C): culture-condition labeller.

Trained ONLY on human corpora (MTC = European folk, POP909 = Chinese pop),
frozen BEFORE any model output is labelled ("冻结先于观察"). Applied later to
LLM outputs; the actual manipulation test (one-sided binomial, BH-corrected)
runs in a later stage on piece-level votes.

Design
------
- Features: token-count vectors over the UNION vocabulary of the two
  corpora; one classifier per alphabet (a1/a2/a3). Cross-alphabet agreement
  is a robustness check mirroring the manuscript's three-encoding rule.
- Model: multinomial Naive Bayes with Laplace smoothing -- deterministic,
  numpy-only, fully reproducible.
- Evaluation: grouped K-fold at the PIECE level (no piece straddles
  train/test; MTC tune-family members are additionally kept together), with
  Wilson 95% CI on piece-level accuracy.
- Inference on model outputs: piece-level majority vote over its windows.

2026-10-23. Stage C of the seven-stage pipeline.
"""
import gzip, json, os, sys, hashlib, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "data_backup", "derived")
RUNS = os.path.join(ROOT, "runs", "c")

MAX_TRAIN_WINDOWS = 20000     # per corpus per alphabet (balanced cap)
ALPHA_SMOOTH = 1.0            # Laplace
SEED = 42
K_FOLD = 5

CORPORA = {"EU": "mtc.windows.jsonl.gz", "CN": "pop909.windows.jsonl.gz"}
FAMILY_KEY = "family"         # MTC tune family: keep members in one fold


def load_windows(fn, ab):
    """-> (windows, group_ids): kept pieces only; group = (corpus, piece)
    with MTC family override so relatives never straddle folds."""
    wins, groups = [], []
    with gzip.open(os.path.join(DATA, fn), "rt", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            if not rec.get("kept"):
                continue
            if rec.get(FAMILY_KEY):
                gid = "fam:" + str(rec[FAMILY_KEY])
            else:
                gid = "pc:" + str(rec.get("id"))
            for w in rec[ab]:
                wins.append(w)
                groups.append(gid)
    return wins, groups


def build_vocab(samples):
    v = collections.Counter()
    for w in samples:
        v.update(w)
    return sorted(v)


def featurize(samples, vocab, index):
    X = np.zeros((len(samples), len(vocab)), dtype=np.int32)
    for r, w in enumerate(samples):
        for t in w:
            j = index.get(t)
            if j is not None:
                X[r, j] += 1
    return X


class MNB:
    def __init__(self):
        self.ll = None
        self.lp = None

    def fit(self, X, y):
        V = X.shape[1]
        ll = np.zeros((2, V))
        for c in (0, 1):
            cnt = X[y == c].sum(axis=0) + ALPHA_SMOOTH
            ll[c] = np.log(cnt / cnt.sum())
        self.ll = ll
        self.lp = np.log(np.bincount(y, minlength=2) / len(y))
        return self

    def predict(self, X):
        s = X @ self.ll.T + self.lp
        return (s[:, 1] > s[:, 0]).astype(int)


def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - m), float(c + m))


def grouped_kfold(X, y, groups):
    """Assign whole groups to folds; return window-level AND piece-level
    (majority vote per group) accuracy counts."""
    rng = np.random.default_rng(SEED)
    gset = sorted(set(groups))
    perm = rng.permutation(len(gset))
    fold_of = {gset[i]: perm[i] % K_FOLD for i in range(len(gset))}
    fold = np.array([fold_of[g] for g in groups])
    w_correct = w_total = 0
    p_correct = p_total = 0
    for k in range(K_FOLD):
        te = fold == k
        m = MNB().fit(X[~te], y[~te])
        pred = m.predict(X[te])
        w_correct += int((pred == y[te]).sum())
        w_total += int(te.sum())
        vote = {}
        for g, t, p in zip(np.array(groups)[te], y[te], pred):
            v = vote.setdefault(g, [0, 0])
            v[0] += int(p == 1)   # CN votes
            v[1] += 1
        for g, (cn, n) in vote.items():
            truth = y[np.array(groups) == g][0]
            p_correct += int((cn * 2 > n) == bool(truth))
            p_total += 1
    return (w_correct, w_total), (p_correct, p_total)


def main():
    os.makedirs(RUNS, exist_ok=True)
    for ab in ("a1", "a2", "a3"):
        Xs, ys, gs = [], [], []
        sizes = {}
        for lab, fn in CORPORA.items():
            wins, groups = load_windows(fn, ab)
            rng = np.random.default_rng(SEED)
            n_take = min(MAX_TRAIN_WINDOWS, len(wins))
            idx = rng.choice(len(wins), size=n_take, replace=False)
            idx.sort()
            Xs.extend(wins[i] for i in idx)
            ys.extend([0 if lab == "EU" else 1] * n_take)
            gs.extend(groups[i] for i in idx)
            sizes[lab] = (len(wins), n_take)
        y = np.array(ys)
        vocab = build_vocab(Xs)
        index = {t: i for i, t in enumerate(vocab)}
        X = featurize(Xs, vocab, index)
        (wc, wn), (pc, pn) = grouped_kfold(X, y, gs)
        wlo, whi = wilson_ci(wc, wn)
        plo, phi = wilson_ci(pc, pn)
        clf = MNB().fit(X, y)
        out = os.path.join(RUNS, f"manip_clf_{ab}.npz")
        np.savez_compressed(
            out, ll=clf.ll, lp=clf.lp, vocab=np.array(vocab), alphabet=ab,
            cv_win_acc=wc / wn, cv_win_lo=wlo, cv_win_hi=whi,
            cv_piece_acc=pc / pn, cv_piece_lo=plo, cv_piece_hi=phi,
            seed=SEED, smooth=ALPHA_SMOOTH,
            train_sizes=json.dumps(sizes))
        print(f"[manip] {ab}: vocab={len(vocab)} "
              f"window-acc={wc}/{wn}={wc/wn:.3f} CI=[{wlo:.3f},{whi:.3f}] | "
              f"piece-vote-acc={pc}/{pn}={pc/pn:.3f} CI=[{plo:.3f},{phi:.3f}]"
              f" -> {os.path.basename(out)}")
    print("[manip] DONE (frozen before labelling any model output)")


if __name__ == '__main__':
    main()
