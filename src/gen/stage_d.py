# -*- coding: utf-8 -*-
"""
Stage D: Proposition P1 -- cross-cultural convergence of LLM-generated music.

Claim: model outputs conditioned on different cultural prompts are MORE similar
to each other than human music from those two cultures is:

    Delta_D = D_JS(pi_CN, pi_EU) - D_JS(Q_CN, Q_EU)      (per model x alphabet)

pi = document-frequency distributions of reference patterns over human pieces
(POP909 = CN, MTC = EU); Q = same over generated pieces. Common culture-fair
pattern space per alphabet.

Protocol amendment 2026-10-23 (documented in docs/experiment_plan.md S4a, rule
frozen BEFORE any short-pattern outcome was inspected):
  PRIMARY space  = union of both corpora' CENSUS patterns of size 2..3
                   (the transmitted combinatorial vocabulary; dense).
  Sensitivity    = 'short123' (sizes 1..3) and 'mfi' (maximal itemsets, the
                   archival fingerprints -- sparse by construction).
  Coverage floor = a cell enters the main Delta_D conclusion only if its mean
                   pattern coverage >= 0.5 patterns/piece; sparser cells are
                   reported separately as decoupling-from-library evidence.

Nulls (frozen): N0 label permutation on humans; N1 on outputs; N2 scale-matched
human resampling at the output cell's piece counts.

Uncertainty: cluster bootstrap (MTC by tune family; POP909 / outputs by piece),
n_boot=1000, percentile 95% CI; big cells (>5k pieces) use n_boot=250 in iid
loops, permutation nulls use 200 draws for pooled sets >3k clusters.

Manipulation-check gate (stage C, majority rule >=2/3 alphabets) annotated per
cell; flagged models are reported but barred from the main conclusion.

Outputs: runs/d/p1_<mode>_results.json (+ .partial per alphabet),
runs/d/df/*.<mode>.npy, runs/d/report_<mode>.txt, caches runs/d/{g,h}cache/.

Usage: python stage_d.py [a1|a2|a3] [short23|short123|mfi]   (default: all, short23)

2026-10-23.
"""
import gzip, json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
DERIVED = WORK / "data_backup" / "derived"
ENC = WORK / "runs" / "c" / "encoded"
OUT = WORK / "runs" / "d"
DFDIR = OUT / "df"; GCACHE = OUT / "gcache"; HCACHE = OUT / "hcache"
MODELS = ["deepseek", "dashscope", "zhipu"]
ALPHS = ["a1", "a2", "a3"]
MODES = {"short23": (2, 3), "short123": (1, 3), "mfi": None}
N_BOOT = 1000
N_BOOT_BIG = 250
N_PERM_BIG = 200
BIG_PIECES = 5000
COVERAGE_FLOOR = 0.5
RNG = np.random.default_rng(20261023)


def log(report, *a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    report.append(s)


# ---------------------------------------------------------------- loading
def load_patterns(alph, mode):
    pats = set()
    for corpus in ("mtc", "pop909"):
        with gzip.open(DERIVED / f"reflib.{corpus}.{alph}.json.gz", "rt", encoding="utf-8") as f:
            rec = json.load(f)
        if mode == "mfi":
            pats.update(tuple(sorted(r[1])) for r in rec["mfi"])
        else:
            lo, hi = MODES[mode]
            items = rec["items"]              # id-string -> token
            for key in rec["census"]:
                ids = key.split("|")
                if lo <= len(ids) <= hi:
                    pats.update([tuple(sorted(items[i] for i in ids))])
    return sorted(pats)


def load_human(alph):
    out = {}
    for corpus in ("mtc", "pop909"):
        pieces, clusters = [], []
        with gzip.open(DERIVED / f"{corpus}.windows.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("kept", True):
                    continue
                pieces.append([frozenset(w) for w in r[alph]])
                clusters.append(r["id"] if corpus == "pop909" else r.get("family", r["id"]))
        out[corpus] = (pieces, clusters)
    return out


def load_outputs(alph):
    out = {}
    for fam in MODELS:
        for cond in ("CN", "EU"):
            pieces, clusters = [], []
            with gzip.open(ENC / f"{fam}_{cond}.windows.jsonl.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    pieces.append([frozenset(w) for w in r[alph]])
                    clusters.append(str(r["i"]))
            out[f"{fam}_{cond}"] = (pieces, clusters)
    return out


# ------------------------------------------------------- hit-matrix build
def build_hits(pieces, space):
    """H[i, j]: piece i contains pattern j in >=1 window. Rarest-token
    inverted index + integer bitmask containment."""
    vocab = sorted({t for p in space for t in p})
    bit = {t: 1 << i for i, t in enumerate(vocab)}
    pmask = []
    tok_patcount = defaultdict(int)
    for p in space:
        m = 0
        for t in p:
            m |= bit[t]
            tok_patcount[t] += 1
        pmask.append(m)
    inv = defaultdict(list)
    for j, p in enumerate(space):
        inv[min(p, key=lambda t: tok_patcount[t])].append(j)
    H = np.zeros((len(pieces), len(space)), dtype=bool)
    for i, wins in enumerate(pieces):
        union = set().union(*wins) if wins else set()
        um = 0
        for t in union:
            b = bit.get(t)
            if b:
                um |= b
        wms = []
        for w in wins:
            m = 0
            for t in w:
                b = bit.get(t)
                if b:
                    m |= b
            wms.append(m)
        for t in union:
            if t not in bit:
                continue
            for j in inv.get(t, ()):
                pm = pmask[j]
                if pm & ~um:
                    continue
                for wm in wms:
                    if (wm & pm) == pm:
                        H[i, j] = True
                        break
    return H


def agg_clusters(H, clusters):
    labels = {}
    ncl = len(set(clusters))
    G = np.zeros((ncl, H.shape[1]), dtype=np.uint16)
    sizes = np.zeros(ncl, dtype=np.uint32)
    for i, c in enumerate(clusters):
        k = labels.get(c)
        if k is None:
            k = labels[c] = len(labels)
        G[k] += H[i]
        sizes[k] += 1
    return G, sizes


# ------------------------------------------------------------- statistics
def norm_rows(C):
    s = C.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return C / s


def js_rows(P, Q):
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(P > 0, P * np.log2(P / M), 0.0).sum(axis=1)
        t2 = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(axis=1)
    return 0.5 * (t1 + t2)


def df_vec(counts, n):
    d = counts.astype(np.float64) / max(n, 1)
    s = d.sum()
    return None if s <= 0 else d / s


def jsdiv(P, Q):
    if P is None or Q is None:
        return float("nan")
    return float(js_rows(P[None, :], Q[None, :])[0])


def multinomial_counts(n_clus, n_boot, rng):
    idx = rng.integers(0, n_clus, size=(n_boot, n_clus), dtype=np.int64)
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * n_clus + idx).ravel()
    C = np.bincount(flat, minlength=n_boot * n_clus).astype(np.float32)
    return C.reshape(n_boot, n_clus)


def boot_gap_agg(G1, s1, G2, s2, n_boot, rng):
    C1 = multinomial_counts(len(s1), n_boot, rng)
    C2 = multinomial_counts(len(s2), n_boot, rng)
    P = norm_rows(C1 @ G1.astype(np.float32))
    Q = norm_rows(C2 @ G2.astype(np.float32))
    return js_rows(P, Q)


def boot_gap_rows(H1, H2, n_boot, rng):
    b = n_boot if max(H1.shape[0], H2.shape[0]) <= BIG_PIECES else N_BOOT_BIG
    gaps = []
    for H in (H1, H2):
        n = H.shape[0]
        if n <= BIG_PIECES:
            C = multinomial_counts(n, b, rng)
            d = norm_rows(C @ H.astype(np.float32))
        else:
            d = np.empty((b, H.shape[1]), dtype=np.float32)
            for r in range(b):
                d[r] = H[rng.integers(0, n, n)].mean(axis=0, dtype=np.float32)
        gaps.append(d)
    return js_rows(gaps[0], gaps[1])


def _split_matrix(n, n1, n_perm, rng):
    """(n_perm, n) 0/1 matrix: each row picks n1 of n items at random."""
    S = np.zeros((n_perm, n), dtype=np.float32)
    for r in range(n_perm):
        S[r, rng.permutation(n)[:n1]] = 1.0
    return S


def perm_gap_agg(G1, G2, n_perm, rng):
    n1 = len(G1)
    G = np.vstack([G1, G2]).astype(np.float32)
    if len(G) > 3000:
        n_perm = N_PERM_BIG
    total = G.sum(axis=0)
    out = np.empty(n_perm)
    B = 200
    for lo in range(0, n_perm, B):
        hi = min(lo + B, n_perm)
        S = _split_matrix(len(G), n1, hi - lo, rng)
        SG = S @ G
        P = norm_rows(SG)
        Q = norm_rows(total[None, :] - SG)
        out[lo:hi] = js_rows(P, Q)
    return out


def perm_gap_rows(H1, H2, n_perm, rng):
    n1 = H1.shape[0]
    H = np.vstack([H1, H2]).astype(np.float32)
    if H.shape[0] > 3000:
        n_perm = N_PERM_BIG
    total = H.sum(axis=0)
    out = np.empty(n_perm)
    B = 200
    for lo in range(0, n_perm, B):
        hi = min(lo + B, n_perm)
        S = _split_matrix(H.shape[0], n1, hi - lo, rng)
        SH = S @ H
        P = norm_rows(SH)
        Q = norm_rows(total[None, :] - SH)
        out[lo:hi] = js_rows(P, Q)
    return out


def scale_matched_human_gap(G1, s1, G2, s2, n1_target, n2_target, n_rep, rng):
    G1f, G2f = G1.astype(np.float32), G2.astype(np.float32)
    # selection matrices: clusters without replacement until ~target pieces
    def pick(sizes, target, b):
        S = np.zeros((b, len(sizes)), dtype=np.float32)
        for r in range(b):
            tot = 0
            for i in rng.permutation(len(sizes)):
                S[r, i] = 1.0
                tot += sizes[i]
                if tot >= target:
                    break
        return S
    out = np.empty(n_rep)
    B = 100
    for lo in range(0, n_rep, B):
        hi = min(lo + B, n_rep)
        b = hi - lo
        S1 = pick(s1, n1_target, b)
        S2 = pick(s2, n2_target, b)
        P = norm_rows(S1 @ G1f)
        Q = norm_rows(S2 @ G2f)
        out[lo:hi] = js_rows(P, Q)
    return out


# ------------------------------------------------------------------ main
def get_human_side(corpus, alph, mode, space, report):
    cg = GCACHE / f"human_{corpus}.{alph}.{mode}.G.npy"
    cs = GCACHE / f"human_{corpus}.{alph}.{mode}.sizes.npy"
    cdf = DFDIR / f"human_{corpus}.{alph}.{mode}.npy"
    if cg.exists():
        G = np.load(cg); sizes = np.load(cs)
        log(report, f"human_{corpus:7s} cache: {int(sizes.sum())} pieces ({len(sizes)} clusters)")
        return G, sizes, np.load(cdf)
    pieces, clusters = load_human(alph)[corpus]
    t = time.time()
    H = build_hits(pieces, space)
    log(report, f"human_{corpus:7s} built: {H.shape[0]} pieces x {H.shape[1]} patterns "
                f"({time.time()-t:.0f}s), coverage {H.sum()/H.shape[0]:.2f} patterns/piece")
    G, sizes = agg_clusters(H, clusters)
    df = H.mean(axis=0)
    np.save(cg, G); np.save(cs, sizes); np.save(cdf, df)
    del H
    return G, sizes, df


def get_output_side(name, alph, mode, space, report):
    ch = HCACHE / f"out_{name}.{alph}.{mode}.npy"
    cdf = DFDIR / f"out_{name}.{alph}.{mode}.npy"
    if ch.exists():
        H = np.load(ch)
        log(report, f"out_{name:16s} cache: {H.shape[0]} pieces")
        return H, np.load(cdf)
    pieces, _ = load_outputs(alph)[name]
    t = time.time()
    H = build_hits(pieces, space)
    log(report, f"out_{name:16s} built: {H.shape[0]} pieces x {H.shape[1]} patterns "
                f"({time.time()-t:.0f}s), coverage {H.sum()/H.shape[0]:.2f} patterns/piece")
    np.save(ch, H)
    df = H.mean(axis=0)
    np.save(cdf, df)
    return H, df


def run_alphabet(alph, mode, models):
    report = []
    log(report, "\n" + "=" * 78)
    log(report, f"STAGE D  P1 convergence   alphabet {alph}   mode {mode}")
    log(report, "=" * 78)
    space = load_patterns(alph, mode)
    log(report, f"pattern space: {len(space)} patterns")
    Gm, sm, df_m = get_human_side("mtc", alph, mode, space, report)
    Gp, sp, df_p = get_human_side("pop909", alph, mode, space, report)

    df_m_v = df_vec(Gm.astype(np.float64).sum(0), sm.sum())
    df_p_v = df_vec(Gp.astype(np.float64).sum(0), sp.sum())
    gap_h = jsdiv(df_p_v, df_m_v)

    sc = OUT / f"shared_{alph}_{mode}.npz"
    if sc.exists():
        z = np.load(sc)
        n0, boot_h = z["n0"], z["boot_h"]
        log(report, "shared nulls/boots loaded from cache")
    else:
        t = time.time()
        n0 = perm_gap_agg(Gp, Gm, N_BOOT if len(Gp) + len(Gm) <= 3000 else N_PERM_BIG, RNG)
        boot_h = boot_gap_agg(Gp, sp, Gm, sm, N_BOOT, RNG)
        np.savez(sc, n0=n0, boot_h=boot_h)
        log(report, f"shared: N0 n={len(n0)}, human boot {N_BOOT} in {time.time()-t:.0f}s")
    log(report, f"human gap D_JS(pi_CN,pi_EU) = {gap_h:.4f}   "
                f"N0 perm p={(n0 >= gap_h).mean():.4f} (n={len(n0)})")

    cov = {"mtc": float(df_m.sum()), "pop909": float(df_p.sum())}
    ares = {"human_gap": gap_h, "human_coverage_per_piece": cov,
            "N0_perm_human": {"mean": float(n0.mean()),
                              "p_above_obs": float((n0 >= gap_h).mean()), "n_perm": len(n0)},
            "models": {}}
    outs = {}
    for fam in models:
        for cond in ("CN", "EU"):
            outs[f"{fam}_{cond}"] = get_output_side(f"{fam}_{cond}", alph, mode, space, report)

    for fam in models:
        Hcn, df_cn = outs[f"{fam}_CN"]; Heu, df_eu = outs[f"{fam}_EU"]
        n_cn, n_eu = Hcn.shape[0], Heu.shape[0]
        c_cn, c_eu = float(df_cn.sum()), float(df_eu.sum())
        eligible = c_cn >= COVERAGE_FLOOR and c_eu >= COVERAGE_FLOOR
        q_cn, q_eu = df_vec(df_cn * n_cn, n_cn), df_vec(df_eu * n_eu, n_eu)
        gap_q = jsdiv(q_cn, q_eu)
        delta = gap_h - gap_q
        t = time.time()
        boot_q = boot_gap_rows(Hcn, Heu, N_BOOT, RNG)
        boot_d = boot_h - boot_q
        ci = np.nanpercentile(boot_d, [2.5, 97.5])
        n1 = perm_gap_rows(Hcn, Heu, N_BOOT, RNG)
        n2 = scale_matched_human_gap(Gm, sm, Gp, sp, n_cn, n_eu, N_BOOT, RNG)
        decomp = {
            "D(Q_CN,pi_CN)": jsdiv(q_cn, df_p_v), "D(Q_CN,pi_EU)": jsdiv(q_cn, df_m_v),
            "D(Q_EU,pi_EU)": jsdiv(q_eu, df_m_v), "D(Q_EU,pi_CN)": jsdiv(q_eu, df_p_v)}
        ares["models"][fam] = {
            "n_pieces": {"CN": n_cn, "EU": n_eu},
            "coverage_per_piece": {"CN": c_cn, "EU": c_eu},
            "eligible_main": bool(eligible),
            "gap_Q": gap_q, "Delta_D": delta,
            "boot_Delta_CI95": [float(ci[0]), float(ci[1])],
            "boot_Delta_p_le0": float((boot_d <= 0).mean()),
            "boot_n": len(boot_q),
            "N1_perm_output": {"mean": float(n1.mean()),
                               "p_above_obs": float((n1 >= gap_q).mean()), "n_perm": len(n1)},
            "N2_scale_matched_human_gap": {
                "mean": float(n2.mean()), "sd": float(n2.std()),
                "p_output_le_human_scaled": float((n2 >= gap_q).mean())},
            "decomposition": decomp,
        }
        m = ares["models"][fam]
        log(report, f"{fam:9s} cov CN={c_cn:.2f} EU={c_eu:.2f} {'MAIN' if eligible else 'sparse'} "
                    f"gap_Q={gap_q:.4f}  Delta_D={delta:+.4f} "
                    f"CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] p(<=0)={(boot_d <= 0).mean():.3f} "
                    f"({time.time()-t:.0f}s)")
        log(report, f"          N1 p={m['N1_perm_output']['p_above_obs']:.4f}  "
                    f"N2 mean={n2.mean():.4f} sd={n2.std():.4f} "
                    f"p(out<=scaled)={m['N2_scale_matched_human_gap']['p_output_le_human_scaled']:.4f}")
        log(report, f"          decomp: Q_CN->pi_CN {decomp['D(Q_CN,pi_CN)']:.4f} "
                    f"Q_CN->pi_EU {decomp['D(Q_CN,pi_EU)']:.4f} | "
                    f"Q_EU->pi_EU {decomp['D(Q_EU,pi_EU)']:.4f} "
                    f"Q_EU->pi_CN {decomp['D(Q_EU,pi_CN)']:.4f}")
    return ares, report


def main():
    t0 = time.time()
    for d in (DFDIR, GCACHE, HCACHE):
        d.mkdir(parents=True, exist_ok=True)
    manip = json.loads((WORK / "runs" / "c" / "manip_results.json").read_text(encoding="utf-8"))
    gate = {}
    for cell in manip["cells"]:
        passes = sum(1 for a in ALPHS if cell["alphabets"][a]["q_BH"] < 0.05)
        gate[f'{cell["family"]}_{cell["condition"]}'] = {
            "n_pass": passes, "pass": passes >= 2,
            "detail": {a: round(cell["alphabets"][a]["frac_culture"], 4) for a in ALPHS}}

    alphs = [a for a in sys.argv[1:] if a in ALPHS] or ALPHS
    modes = [m for m in sys.argv[1:] if m in MODES] or ["short23"]
    models = [m for m in sys.argv[1:] if m in MODELS] or MODELS
    mode = modes[0]
    res_path = OUT / f"p1_{mode}_results.json"
    results = {"mode": mode, "coverage_floor": COVERAGE_FLOOR,
               "n_boot": N_BOOT, "n_boot_big": N_BOOT_BIG, "seed": 20261023,
               "gate": gate, "alphabets": {}, "elapsed_sec": None}
    if res_path.exists():
        try:
            results["alphabets"] = json.loads(res_path.read_text(encoding="utf-8"))["alphabets"]
        except Exception:
            pass
    all_report = [f"STAGE D  P1 convergence  mode {mode}  (2026-10-23)"]
    rpt = OUT / f"report_{mode}.txt"
    if rpt.exists():
        all_report = rpt.read_text(encoding="utf-8").splitlines()
    for alph in alphs:
        ares, rep = run_alphabet(alph, mode, models)
        results["alphabets"].setdefault(alph, {"models": {}})
        results["alphabets"][alph].update({k: v for k, v in ares.items() if k != "models"})
        results["alphabets"][alph]["models"].update(ares["models"])
        all_report += rep
        (OUT / f"p1_{mode}_results.{alph}.partial.json").write_text(
            json.dumps({alph: ares}, indent=1), encoding="utf-8")
        (OUT / f"report_{mode}.txt").write_text("\n".join(all_report), encoding="utf-8")
        res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    res_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(all_report, f"\n[saved] {res_path}  elapsed {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
