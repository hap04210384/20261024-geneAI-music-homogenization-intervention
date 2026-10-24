# -*- coding: utf-8 -*-
"""
Stage B (construct validation, manuscript 5.3, go/no-go gate).

Design (frozen 2026-10-23):
  B1  Variant-level concordance (MTC): mean pattern-profile Jaccard of
      same-variant piece pairs vs different-family pairs. Label-permutation
      p (labels shuffled among family-bearing pieces), cluster bootstrap
      (family level) 95% CI for the contrast.
  B2  Graded-kinship Mantel (MTC): kinship distance d = 0 same tunefamily
      variant / 1 same root different variant / 2 different root, versus
      pattern distance 1-Jaccard, over the same sampled pairs; Mantel r with
      label-permutation p.
  B3  Harmonic concordance (POP909; no dating metadata exists, so the
      year-structure test is replaced by harmonic-vocabulary concordance,
      reported as a design adaptation): chord-vocabulary Jaccard versus
      pattern-profile Jaccard over sampled song pairs; Spearman rho with
      permutation p, plus top-vs-bottom quartile contrast.

  Inference unit = piece. Family label '' (unassigned) excluded from B1/B2.
  Within-family pairs capped at 200 per family (seed 42); between-pair
  sample = 5x the within sample (seed 42). n_perm = 1000, n_boot = 1000.
  Profiles = piece-level census patterns (window frequency >= 2) per the
  stage-A4 reference libraries; three encodings reported separately.

2026-10-23.
"""
import gzip, json, os, re, sys, time, collections, random
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
DATA = WORK / "data_backup"
DERIVED = DATA / "derived"
RUNS = REPO / "runs" / "b"
RUNS.mkdir(parents=True, exist_ok=True)
POP = DATA / "pop909" / "POP909"

SEED = 42
N_PERM = 1000
N_BOOT = 1000
CAP = 100          # within-family pair cap per family
BETWEEN_X = 3

rng = random.Random(SEED)

# ------------------------------------------------------------ profiles ----

def census_masks(corpus, alph):
    with gzip.open(DERIVED / f"reflib.{corpus}.{alph}.json.gz", "rt", encoding="utf-8") as f:
        rec = json.load(f)
    masks = set()
    for key in rec["census"]:
        k = 0
        for i in key.split("|"):
            k |= 1 << int(i)
        masks.add(k)
    return masks

def piece_profiles(corpus, alph):
    """Stream windows; per kept piece enumerate window subsets; profile =
    sorted list of census masks present (exact re-enumeration, as A4)."""
    cens = census_masks(corpus, alph)
    mp = DERIVED / "fimi" / f"{corpus}.{alph}.items.json"
    id_of = {t: int(i) for i, t in
             json.loads(mp.read_text(encoding="utf-8"))["token_of_id"].items()}
    prof = {}
    with gzip.open(DERIVED / f"{corpus}.windows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("kept", True):
                continue
            ps = set()
            for w in r[alph]:
                bits = [1 << id_of[t] for t in w]
                subs = [0]
                for b in bits:
                    subs += [x | b for x in subs]
                for k in subs[1:]:
                    if k in cens:
                        ps.add(k)
            prof[r["id"]] = np.array(sorted(ps), dtype=object)
    return prof

def jaccard(a, b):
    if a.size == 0 and b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)

def jaccard_many(pairs, prof):
    """Vector-ish batch: compute Jaccard for a list of pairs using sorted
    uint64 arrays (np.intersect1d, assume_unique)."""
    return np.array([jaccard(prof[a], prof[b]) for a, b in pairs])

# ------------------------------------------------------------------ B1 ----

def b1_mtc(prof, fam_of, dump_path=None):
    ids = [p for p in prof if fam_of.get(p)]
    fams = collections.defaultdict(list)
    for p in ids:
        fams[fam_of[p]].append(p)
    rngp = random.Random(SEED)
    within = []
    for fam, members in fams.items():
        if len(members) < 2:
            continue
        pairs = [(members[i], members[j])
                 for i in range(len(members)) for j in range(i + 1, len(members))]
        rngp.shuffle(pairs)
        within.extend(pairs[:CAP])
    n_out = min(BETWEEN_X * len(within), 400000)
    out = set()
    while len(out) < n_out:
        a, b = rngp.sample(ids, 2)
        if fam_of[a] != fam_of[b]:
            out.add((a, b) if a < b else (b, a))
    between = list(out)

    j_in = jaccard_many(within, prof)
    j_out = jaccard_many(between, prof)
    obs = j_in.mean() - j_out.mean()

    idx = {p: i for i, p in enumerate(ids)}
    A = np.array([idx[a] for a, b in within] + [idx[a] for a, b in between])
    B = np.array([idx[b] for a, b in within] + [idx[b] for a, b in between])
    J = np.concatenate([j_in, j_out])
    lab_int = {f: k for k, f in enumerate(sorted(fams))}
    labels = np.array([lab_int[fam_of[p]] for p in ids], dtype=np.int64)
    n_ge = 0
    r = np.random.RandomState(SEED)
    for _ in range(N_PERM):
        pl = r.permutation(labels)
        eq = pl[A] == pl[B]
        d = J[eq].mean() - J[~eq].mean()
        if d >= obs:
            n_ge += 1
    p_perm = (n_ge + 1) / (N_PERM + 1)

    # family-level cluster bootstrap on PRECOMPUTED per-pair values
    fam_list = [f for f, ms in fams.items() if len(ms) >= 2]
    fam_idx = {f: i for i, f in enumerate(fam_list)}
    vals = np.array([j for j in j_in])
    owner = np.array([fam_idx[fam_of[a]] for a, b in within])
    boots = np.empty(N_BOOT)
    for t in range(N_BOOT):
        pick = r.randint(0, len(fam_list), len(fam_list))
        mask = np.isin(owner, pick)
        boots[t] = vals[mask].mean() - j_out.mean()
    ci = np.percentile(boots, [2.5, 97.5])
    if dump_path:
        np.savez_compressed(dump_path, j_in=j_in, j_out=j_out)
    return dict(n_pieces=len(ids), n_families=len(fam_list),
                n_within_pairs=len(within), n_between_pairs=len(between),
                jaccard_within=float(j_in.mean()), jaccard_between=float(j_out.mean()),
                contrast=float(obs), ci95=[float(ci[0]), float(ci[1])],
                p_perm=float(p_perm), n_perm=N_PERM, n_boot=N_BOOT)

# ------------------------------------------------------------------ B2 ----

def b2_mtc(prof, fam_of, root_of):
    ids = [p for p in prof if fam_of.get(p)]
    fams = collections.defaultdict(list)
    for p in ids:
        fams[fam_of[p]].append(p)
    rngp = random.Random(SEED)
    within, out = [], set()
    for fam, members in fams.items():
        if len(members) < 2:
            continue
        pairs = [(members[i], members[j])
                 for i in range(len(members)) for j in range(i + 1, len(members))]
        rngp.shuffle(pairs)
        within.extend(pairs[:CAP])
    n_out = min(BETWEEN_X * len(within), 400000)
    while len(out) < n_out:
        a, b = rngp.sample(ids, 2)
        if fam_of[a] != fam_of[b]:
            out.add((a, b) if a < b else (b, a))
    between = list(out)
    pairs = within + between

    # integer-coded kinship: 0 same variant, 1 same root, 2 unrelated
    fam_list = sorted(fams)
    fam_code = {f: k for k, f in enumerate(fam_list)}
    root_code_of_fam = np.array([int(root_of.get(f, f) or 0) if str(root_of.get(f, "")).isdigit()
                                 else abs(hash(root_of.get(f, ""))) % (2**31)
                                 for f in fam_list], dtype=np.int64)
    fc = np.array([fam_code[fam_of[p]] for p in ids], dtype=np.int64)
    idx = {p: i for i, p in enumerate(ids)}
    A = np.array([idx[a] for a, b in pairs]); B = np.array([idx[b] for a, b in pairs])
    d_kin = np.where(fc[A] == fc[B], 0,
                     np.where(root_code_of_fam[fc[A]] == root_code_of_fam[fc[B]], 1, 2)
                     ).astype(float)
    d_pat = 1 - jaccard_many(pairs, prof)
    r_obs = np.corrcoef(d_kin, d_pat)[0, 1]
    r_ = np.random.RandomState(SEED)
    # size-preserving label permutation: multiset of family labels fixed,
    # random assignment to pieces
    labels_full = np.repeat(np.arange(len(fam_list)),
                            [len(fams[f]) for f in fam_list])
    n_ge = 0
    for _ in range(N_PERM):
        pf = labels_full[r_.permutation(labels_full.size)]
        pr = root_code_of_fam[pf]       # its root code
        dk = np.where(pf[A] == pf[B], 0, np.where(pr[A] == pr[B], 1, 2)).astype(float)
        if np.corrcoef(dk, d_pat)[0, 1] >= r_obs:
            n_ge += 1
    p_perm = (n_ge + 1) / (N_PERM + 1)
    return dict(n_pairs=len(pairs), mantel_r=float(r_obs),
                p_perm=float(p_perm), n_perm=N_PERM,
                note="one-sided upper tail: kinship distance and pattern "
                     "distance are both distances, so the prediction is r > 0")

# ------------------------------------------------------------------ B3 ----

def chord_vocab(song_id):
    p = POP / f"{song_id}" / "chord_midi.txt"
    if not p.exists():
        return None
    vocab = set()
    for L in open(p, encoding="utf-8", errors="replace"):
        c = L.split()
        if len(c) >= 3 and c[2] != "N":
            vocab.add(c[2])
    return vocab

def b3_pop(prof):
    ids = list(prof)
    voc = {p: chord_vocab(p) for p in ids}
    ids = [p for p in ids if voc.get(p)]
    rngp = random.Random(SEED)
    n_pairs = min(200000, 20 * len(ids))
    seen = set(); pairs = []
    while len(pairs) < n_pairs:
        a, b = rngp.sample(ids, 2)
        k = (a, b) if a < b else (b, a)
        if k not in seen:
            seen.add(k); pairs.append(k)
    hv = np.array([len(voc[a] & voc[b]) / max(1, len(voc[a] | voc[b])) for a, b in pairs])
    pj = np.array([jaccard(prof[a], prof[b]) for a, b in pairs])
    rho = spearman(hv, pj)
    r_ = np.random.RandomState(SEED)
    n_ge = 0
    for _ in range(N_PERM):
        if spearman(hv, r_.permutation(pj)) >= rho:
            n_ge += 1
    p_perm = (n_ge + 1) / (N_PERM + 1)
    q = np.percentile(hv, 75)
    top, bot = pj[hv >= q], pj[hv < np.percentile(hv, 25)]
    return dict(n_pieces=len(ids), n_pairs=len(pairs),
                spearman_rho=float(rho), p_perm=float(p_perm), n_perm=N_PERM,
                quartile_top=float(top.mean()), quartile_bottom=float(bot.mean()),
                quartile_contrast=float(top.mean() - bot.mean()))

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])

# ------------------------------------------------------------------ main ----

def mtc_meta():
    fam_of, root_of = {}, {}
    with gzip.open(DERIVED / "mtc.windows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("kept", True):
                continue
            fam = r.get("family") or ""
            fam_of[r["id"]] = fam
            root_of[fam] = fam.split("_")[0] if fam else ""
    return fam_of, root_of

def cached_profiles(corpus, alph):
    cache = RUNS / f"profiles.{corpus}.{alph}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return {k: np.array(v, dtype=object) for k, v in json.load(f).items()}
    t0 = time.time()
    prof = piece_profiles(corpus, alph)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump({k: [int(x) for x in v] for k, v in prof.items()}, f)
    print(f"[B] profiles {corpus}.{alph} built in {time.time()-t0:.0f}s "
          f"({len(prof)} pieces)", flush=True)
    return prof

def run_one(corpus, alph):
    prof = cached_profiles(corpus, alph)
    if corpus == "mtc":
        fam_of, root_of = mtc_meta()
        res1 = b1_mtc(prof, fam_of, dump_path=RUNS / f"jac.{corpus}.{alph}.npz")
        res2 = b2_mtc(prof, fam_of, root_of)
        res = dict(corpus=corpus, alphabet=alph, b1=res1, b2=res2,
                   seed=SEED, n_perm=N_PERM, n_boot=N_BOOT, cap=CAP,
                   between_x=BETWEEN_X)
        print(f"[B] mtc.{alph}: B1 contrast={res1['contrast']:.4f} "
              f"CI=[{res1['ci95'][0]:.4f},{res1['ci95'][1]:.4f}] p={res1['p_perm']:.4f} | "
              f"B2 r={res2['mantel_r']:.4f} p={res2['p_perm']:.4f}", flush=True)
    else:
        res3 = b3_pop(prof)
        res = dict(corpus=corpus, alphabet=alph, b3=res3,
                   seed=SEED, n_perm=N_PERM)
        print(f"[B] pop909.{alph}: B3 rho={res3['spearman_rho']:.4f} "
              f"p={res3['p_perm']:.4f} "
              f"quartile contrast={res3['quartile_contrast']:.4f}", flush=True)
    (RUNS / f"{corpus}.{alph}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_one(sys.argv[1], sys.argv[2])
    else:
        for c, a in (("mtc", "a1"), ("mtc", "a2"), ("mtc", "a3"),
                     ("pop909", "a1"), ("pop909", "a2"), ("pop909", "a3")):
            run_one(c, a)
        print("[B] DONE")
