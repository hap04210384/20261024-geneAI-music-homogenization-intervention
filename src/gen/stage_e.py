# -*- coding: utf-8 -*-
"""
Stage E: Proposition P2 -- structured collapse of the transmitted pattern
vocabulary. THE sharpest pre-registered test of the bottleneck account.

Frozen protocol (decided 2026-10-24 BEFORE inspecting any output-side result;
strata/thresholds are computed from the HUMAN reference libraries only and
saved before any output statistic is touched):

  Space          = stage-D primary space per alphabet (census sizes 2..3,
                   union of both libraries; culture-fair).
  Counts         c_X(p) = pattern piece-count in corpus X (POP909 = CN,
                   MTC = EU). df = c / n_pieces.
  Mass           m(p) = c_CN(p) + c_EU(p).
  Strata         S1/S2/S3 = top/middle/bottom tertile of m(p) over the space
                   (S1 = highest-frequency patterns). Frozen from humans.
  Specificity    prevalence-based (per-piece, corpus-size independent):
                 spec(p) = |ln((df_CN+eps_CN)/(df_EU+eps_EU))| with
                 eps_X = 0.5/n_X (half a pseudo-piece); df = document
                 frequency. Highly culture-specific = {spec >= tau90};
                 neutral = {spec <= median(spec)}. Frozen from humans.
                 (Amendment E-3, 2026-10-24: E-1 used this same formula but
                 conditioned the tail population on the prompted culture
                 only, which emptied the neutral class; E-2 fixed the
                 population to both-present patterns; E-2's share-based
                 spec is discarded because corpus coverage asymmetry
                 (~40x total-mass ratio) made any both-present rare pattern
                 look culture-specific by construction.)
  Survival       s(p) = df_Q(p) / df_pi_prompted(p), defined where the
                   prompted-culture reference df > 0; df_Q = 0 counts as 0.
  Stratum surv.  sigma(S_k) = sum_{p in S_k, pi>0} m(p) s(p) / sum m(p)
                   (mass-weighted; mass frozen from humans).
  Entropy        H = -sum x log2(x) over the normalised df vector.

Pre-registered tests (per model x condition; alphabet a3 primary):
  T1  H(Q) < H(pi_prompted)            -- diversity collapse
  T2  sigma(S1) > sigma(S2) > sigma(S3) -- ordered survival, diffs + 95% CI
  T3  within the non-core shared vocabulary (both-present patterns in
      S2+S3; amendment E-4 2026-10-24: S3-only is not estimable because
      cross-culture overlap is structurally concentrated in heavier
      patterns -- verified on human side only, neutral class empty in
      S3 at a1/a2). AMENDMENT E-5 (same day, after a1/a2 diagnostics
      showed a POSITIVE raw spec-survival association confounded by
      pattern mass): the high-vs-neutral comparison is MASS-BAND
      MATCHED -- the population is split into mass terciles, survival
      is compared within each band, and bands are combined by direct
      standardisation over the pooled mass distribution. Reported
      alongside: raw comparison and Spearman(spec, s) for reference.
      + T3b matching- vs mismatched-prompt survival of high-spec
      patterns. (Population definitions use frozen human quantities
      only; bands are also computed from frozen human mass.)
Kill-theory point: if T2's ordering fails, the bottleneck account's structural
prediction is falsified.

Uncertainty: piece-level bootstrap over OUTPUT pieces only (the reference
libraries are frozen point estimates), n_boot=1000, percentile 95% CI, GEMM
batched. Manipulation-check gate annotation identical to stage D.

Outputs: runs/e/strata_<alph>.json (frozen BEFORE outputs), runs/e/p2_results.json,
runs/e/report.txt.

Usage: python stage_e.py [a1|a2|a3] [model ...]   (default: all, all models)

2026-10-24.
"""
import gzip, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
OUTD = WORK / "runs" / "d"          # stage-D caches (df vectors + hit matrices)
OUTE = WORK / "runs" / "e"
MODELS = ["deepseek", "dashscope", "zhipu"]
ALPHS = ["a1", "a2", "a3"]
MODE = "short23"
N_BOOT = 1000
RNG = np.random.default_rng(20261024)


def log(report, *a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    report.append(s)


def entropy(dist):
    d = dist[dist > 0]
    return float(-(d * np.log2(d)).sum())


def boot_df(H, n_boot, rng):
    """(n_boot, p) resampled document-frequency matrices via GEMM."""
    n = H.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * n + idx).ravel()
    C = np.bincount(flat, minlength=n_boot * n).astype(np.float32)
    return C.reshape(n_boot, n) @ H.astype(np.float32) / n


def sigma_rows(dfrows, cols, m, pi):
    """Mass-weighted survival per bootstrap row. dfrows (b,p) output dfs;
    cols = indices of patterns in stratum with pi>0; m = frozen mass on cols
    (normalised); pi = prompted-culture reference df vector. Returns (b,)."""
    s = dfrows[:, cols] / pi[cols][None, :]
    return s @ m


def spearman_col(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return 0.0 if d == 0 else float((ra * rb).sum() / d)


def run_alphabet(alph, models, report):
    log(report, "\n" + "=" * 78)
    log(report, f"STAGE E  P2 structured collapse   alphabet {alph}   mode {MODE}")
    log(report, "=" * 78)
    t0 = time.time()

    # ---------------- frozen strata from HUMAN libraries only ----------------
    spath = OUTE / f"strata_{alph}.json"
    df_cn = np.load(OUTD / "df" / f"human_pop909.{alph}.{MODE}.npy")   # pi_CN
    df_eu = np.load(OUTD / "df" / f"human_mtc.{alph}.{MODE}.npy")      # pi_EU
    n_cn, n_eu = 908, 17638
    c_cn, c_eu = df_cn * n_cn, df_eu * n_eu
    mass = c_cn + c_eu

    def specificity(d1, d2, n1, n2):
        """Prevalence-based cultural specificity from document frequencies
        (corpus-size independent; half a pseudo-piece smoothing)."""
        return np.abs(np.log((d1 + 0.5 / n1) / (d2 + 0.5 / n2)))

    spec = specificity(df_cn, df_eu, n_cn, n_eu)
    if spath.exists():
        fr = json.loads(spath.read_text(encoding="utf-8"))
        log(report, f"frozen strata loaded: {spath.name}")
    else:
        tau = float(np.percentile(spec, 90))
        neutral = float(np.percentile(spec, 50))
        order = np.argsort(-mass)                     # descending mass
        tert = len(order) // 3
        S = {"S1": order[:tert].tolist(), "S2": order[tert:2 * tert].tolist(),
             "S3": order[2 * tert:].tolist()}
        fr = {"alphabet": alph, "mode": MODE, "n_patterns": len(mass),
              "spec_tau90": tau, "spec_median": neutral,
              "mass_tertile_cutoffs": [float(mass[order[tert]]),
                                       float(mass[order[2 * tert]])],
              "S": S,
              "frozen_from": "human reference libraries only (POP909 + MTC)",
              "frozen_at": "2026-10-24 (before any output-side statistic)",
              "spec_definition": "prevalence-based |ln((df_CN+0.5/n_CN)/(df_EU+0.5/n_EU))|"}
        spath.write_text(json.dumps(fr), encoding="utf-8")
        log(report, f"[FROZEN] strata saved: {spath.name}  tau90={tau:.3f} "
                    f"median={neutral:.3f}  |S1|={tert} |S2|={len(S['S2'])} |S3|={len(S['S3'])}")
    spec_tau = fr["spec_tau90"]
    spec_med = fr["spec_median"]
    strata = {k: np.asarray(v, dtype=np.int64) for k, v in fr["S"].items()}

    # per-condition reference df vectors (frozen)
    pi = {"CN": df_cn, "EU": df_eu}
    Hpi = {"CN": entropy(df_cn / df_cn.sum()), "EU": entropy(df_eu / df_eu.sum())}
    log(report, f"H(pi_CN)={Hpi['CN']:.3f} bits  H(pi_EU)={Hpi['EU']:.3f} bits")

    ares = {"strata_ref": str(spath), "H_pi": Hpi, "models": {}}
    for fam in models:
        for cond in ("CN", "EU"):
            H = np.load(OUTD / "hcache" / f"out_{fam}_{cond}.{alph}.{MODE}.npy")
            n = H.shape[0]
            df_q = H.mean(axis=0)
            t = time.time()
            boot = boot_df(H, N_BOOT, RNG)            # (b, p) resampled dfs
            # T1 entropy
            ent = np.array([entropy(r / r.sum()) for r in boot])
            ent_ci = np.percentile(ent, [2.5, 97.5])
            t1_p = float((ent >= Hpi[cond]).mean())
            # T2 ordered survival
            sig = {}
            sig_boot = {}
            ref = pi[cond]
            for k in ("S1", "S2", "S3"):
                cols = strata[k][ref[strata[k]] > 0]
                m = mass[cols] / mass[cols].sum()
                s_obs = float((m * (df_q[cols] / ref[cols])).sum())
                sig[k] = s_obs
                sig_boot[k] = sigma_rows(boot, cols, m, ref)
            d12 = sig_boot["S1"] - sig_boot["S2"]
            d23 = sig_boot["S2"] - sig_boot["S3"]
            ci12 = np.percentile(d12, [2.5, 97.5]); ci23 = np.percentile(d23, [2.5, 97.5])
            p12 = float((d12 <= 0).mean()); p23 = float((d23 <= 0).mean())
            # T3 tail tests within the non-core shared vocabulary
            # (both-present patterns in S2+S3; amendment E-4).
            cols_tail = np.concatenate([strata["S2"], strata["S3"]])
            other_pi = pi["CN"] if cond == "EU" else pi["EU"]
            cols3 = cols_tail[(ref[cols_tail] > 0) & (other_pi[cols_tail] > 0)]
            high = cols3[spec[cols3] >= spec_tau]
            neut = cols3[spec[cols3] <= spec_med]
            def mws(cols):
                m = mass[cols] / mass[cols].sum()
                obs = float((m * (df_q[cols] / ref[cols])).sum())
                b = sigma_rows(boot, cols, m, ref)
                return obs, b
            s_high, b_high = mws(high) if len(high) else (float("nan"), None)
            s_neut, b_neut = mws(neut) if len(neut) else (float("nan"), None)
            if b_high is not None and b_neut is not None:
                dh = b_high - b_neut
                ci_h = [float(x) for x in np.percentile(dh, [2.5, 97.5])]
                p_h = float((dh >= 0).mean())
            else:
                ci_h, p_h = [None, None], None
            # E-5: mass-band matched comparison within the T3 population
            mb = mass[cols3]
            bnds = np.digitize(mb, np.percentile(mb, [100 / 3, 200 / 3]))
            band_res = {}
            diff_boot = np.zeros(N_BOOT)
            for bnd in range(3):
                sub = cols3[bnds == bnd]
                hi_b = sub[spec[sub] >= spec_tau]
                ne_b = sub[spec[sub] <= spec_med]
                if len(hi_b) == 0 or len(ne_b) == 0:
                    band_res[f"band{bnd}"] = {"n_high": len(hi_b), "n_neutral": len(ne_b)}
                    continue
                w = float(mass[sub].sum() / mass[cols3].sum())
                sh, bh = mws(hi_b)
                sn, bn = mws(ne_b)
                d = bh - bn
                band_res[f"band{bnd}"] = {
                    "n_high": len(hi_b), "n_neutral": len(ne_b), "weight": w,
                    "sigma_high": sh, "sigma_neutral": sn,
                    "CI_diff": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                    "p_diff_ge0": float((d >= 0).mean())}
                diff_boot += w * d
            ci_adj = [float(x) for x in np.percentile(diff_boot, [2.5, 97.5])]
            p_adj = float((diff_boot >= 0).mean())
            # T3b: matching- vs mismatched-culture prompt survival for
            # high-specificity patterns (present in both cultures)
            other = "EU" if cond == "CN" else "CN"
            Hoth = np.load(OUTD / "hcache" / f"out_{fam}_{other}.{alph}.{MODE}.npy")
            df_o = Hoth.mean(axis=0)
            if len(high):
                mo = mass[high] / mass[high].sum()
                s_match = float((mo * (df_q[high] / ref[high])).sum())
                s_mism = float((mo * (df_o[high] / ref[high])).sum())
            else:
                s_match = s_mism = float("nan")
            # Spearman spec vs survival within S3 both-present
            surv3 = df_q[cols3] / ref[cols3]
            rho = spearman_col(spec[cols3], surv3) if len(cols3) > 2 else float("nan")
            ares["models"].setdefault(fam, {})[cond] = {
                "n_pieces": n,
                "T1_entropy": {"H_Q": float(entropy(df_q / df_q.sum())),
                               "H_pi": Hpi[cond],
                               "boot_CI95": [float(ent_ci[0]), float(ent_ci[1])],
                               "p_HQ_ge_Hpi": t1_p},
                "T2_survival": {k: sig[k] for k in sig},
                "T2_diffs": {"S1_minus_S2": [float(ci12[0]), float(ci12[1])], "p": p12},
                "T2_diffs_23": {"S2_minus_S3": [float(ci23[0]), float(ci23[1])], "p": p23},
                "T3_tail": {"sigma_high": s_high, "sigma_neutral": s_neut,
                            "CI_high_minus_neutral": ci_h, "p": p_h,
                            "n_high": len(high), "n_neutral": len(neut),
                            "n_population": len(cols3),
                            "spearman_spec_survival": rho,
                            "massband_adjusted": {"bands": band_res,
                                                  "CI_diff": ci_adj, "p_diff_ge0": p_adj}},
                "T3b_match": {"sigma_matching_prompt": s_match,
                              "sigma_mismatched_prompt": s_mism},
            }
            a = ares["models"][fam][cond]
            log(report, f"{fam:9s} {cond}  H(Q)={a['T1_entropy']['H_Q']:.2f} vs "
                        f"H(pi)={Hpi[cond]:.2f} p={t1_p:.3f} | "
                        f"sigma S1={sig['S1']:.3f} S2={sig['S2']:.3f} S3={sig['S3']:.3f} "
                        f"d12 CI=[{ci12[0]:+.3f},{ci12[1]:+.3f}] p={p12:.3f} "
                        f"d23 CI=[{ci23[0]:+.3f},{ci23[1]:+.3f}] p={p23:.3f} | "
                        f"tail(n={len(cols3)}) adjCI=[{ci_adj[0]:+.3f},{ci_adj[1]:+.3f}] "
                        f"p={p_adj:.3f} rho={rho:.3f} | "
                        f"match={s_match:.3f} mism={s_mism:.3f} ({time.time()-t:.0f}s)")
    return ares, report, time.time() - t0


def _json_default(o):
    import numpy as _np
    if isinstance(o, _np.ndarray):
        print(f"[json-fix] ndarray at dump: shape={o.shape} dtype={o.dtype}", flush=True)
        return o.tolist()
    if isinstance(o, _np.generic):
        print(f"[json-fix] np scalar: {o.dtype}", flush=True)
        return o.item()
    raise TypeError(f"not serializable: {type(o)}")


def main():
    t0 = time.time()
    OUTE.mkdir(parents=True, exist_ok=True)
    manip = json.loads((WORK / "runs" / "c" / "manip_results.json").read_text(encoding="utf-8"))
    gate = {}
    for cell in manip["cells"]:
        passes = sum(1 for a in ALPHS if cell["alphabets"][a]["q_BH"] < 0.05)
        gate[f'{cell["family"]}_{cell["condition"]}'] = {
            "n_pass": passes, "pass": passes >= 2,
            "a3_pass": cell["alphabets"]["a3"]["q_BH"] < 0.05}

    alphs = [a for a in sys.argv[1:] if a in ALPHS] or ALPHS
    models = [m for m in sys.argv[1:] if m in MODELS] or MODELS
    results = {"mode": MODE, "n_boot": N_BOOT, "seed": 20261024, "gate": gate,
               "alphabets": {}, "elapsed_sec": None}
    res_path = OUTE / "p2_results.json"
    if res_path.exists():
        try:
            results["alphabets"] = json.loads(res_path.read_text(encoding="utf-8"))["alphabets"]
        except Exception:
            pass
    all_report = [f"STAGE E  P2 structured collapse   (2026-10-24)"]
    rpt = OUTE / "report.txt"
    if rpt.exists():
        all_report = rpt.read_text(encoding="utf-8").splitlines()
    for alph in alphs:
        ares, rep, el = run_alphabet(alph, models, all_report)
        results["alphabets"][alph] = ares
        (OUTE / "report.txt").write_text("\n".join(all_report), encoding="utf-8")
        res_path.write_text(json.dumps(results, indent=1, default=_json_default), encoding="utf-8")
    results["elapsed_sec"] = round(time.time() - t0, 1)
    res_path.write_text(json.dumps(results, indent=1, default=_json_default), encoding="utf-8")
    log(all_report, f"\n[saved] {res_path}  elapsed {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
