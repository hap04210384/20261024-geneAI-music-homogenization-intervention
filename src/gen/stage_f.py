# -*- coding: utf-8 -*-
"""Stage F: Proposition P3 -- dose-response of cultural fidelity to seed injection.

Restorable-transmission test.  For each model x mechanism x arm x culture:

    F(c) = 1 - D_JS( df(cell at dose c) || pi_prompted )      (a3, short23 space)

Mechanisms (frozen in docs/experiment_plan.md S6a, amendment F-1):
  M1  continuation priming (own-culture seed opening prepended, model continues)
  M2  notated injection    (seed list in prompt; own and cross-cultural arms)
  M3  output-side filter   (control, ZERO new sampling; stage-C pool ranked by
                            coverage of the dose-c seed pattern set, top
                            max(10, ceil((1-c)*N)) pieces retained -- amendment
                            F-3, analysis rule fixed before any P3 outcome was
                            inspected; parametrised so c has the same
                            "amount of cultural material enforced" meaning as in
                            M1/M2 and curves share direction & dose axis)

Doses: c in {0, 0.1, 0.25, 0.5, 1.0}; c = 0 reuses the stage-C unseeded pool.

Trend per series: Kendall tau (5 doses, sign concordance), gain = F(1)-F(0)
with cluster(piece) bootstrap CI95 (n_boot=1000, multinomial trick as in
stage D), saturation = F(1)-F(0.5).
Cross-arm criterion (M2): per dose, F_own - F_cross with bootstrap CI95.

Outputs: runs/f/p3_results.json, runs/f/report_f.txt,
         caches runs/f/hcache/<cell>.a3.short23.npy

Usage: python stage_f.py

2026-10-24.
"""
import gzip, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_d as D

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parent
RUNF = WORK / "runs" / "f"
RUND = WORK / "runs" / "d"
ENC_F = RUNF / "encoded"
HCACHE = RUNF / "hcache"; HCACHE.mkdir(parents=True, exist_ok=True)
MODELS = ["deepseek", "dashscope", "zhipu"]
DOSES = [0.0, 0.1, 0.25, 0.5, 1.0]
CDOSES = [0.1, 0.25, 0.5, 1.0]
N_BOOT = 1000
MIN_KEEP = 10
RNG = np.random.default_rng(20261024)
ALPH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in D.ALPHS else "a3"
MODE = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in D.MODES else "short23"
_DEFAULT = (ALPH, MODE) == ("a3", "short23")
RES_JSON = RUNF / ("p3_results.json" if _DEFAULT else f"p3_{ALPH}.{MODE}_results.json")
RES_RPT = RUNF / ("report_f.txt" if _DEFAULT else f"report_{ALPH}.{MODE}.txt")


def save_results(results, report):
    RES_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")
    RES_RPT.write_text("\n".join(report), encoding="utf-8")


def log(report, *a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    report.append(s)


def norm_df(raw):
    d = raw.astype(np.float64)
    s = d.sum()
    return None if s <= 0 else d / s


def load_pi(cond):
    corpus = "pop909" if cond == "CN" else "mtc"
    return norm_df(np.load(RUND / "df" / f"human_{corpus}.{ALPH}.{MODE}.npy"))


def load_stageC_H(fam, cond):
    return np.load(RUND / "hcache" / f"out_{fam}_{cond}.{ALPH}.{MODE}.npy")


def cell_H(cell, space):
    ch = HCACHE / f"{cell}.{ALPH}.{MODE}.npy"
    if ch.exists():
        return np.load(ch)
    pieces = []
    with gzip.open(ENC_F / f"{cell}.windows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pieces.append([frozenset(w) for w in r[ALPH]])
    H = D.build_hits(pieces, space)
    np.save(ch, H)
    return H


def boot_F(H, pi, n_boot, rng):
    """Bootstrap distribution of F = 1 - D_JS(norm(df_resampled) || pi)."""
    n = H.shape[0]
    C = D.multinomial_counts(n, n_boot, rng)
    d = D.norm_rows(C @ H.astype(np.float32))
    return 1.0 - D.js_rows(d, np.tile(pi, (n_boot, 1)))


def kendall_tau(xs, ys):
    n = len(xs)
    c = d = t = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(xs[j] - xs[i])
            sy = np.sign(ys[j] - ys[i])
            if sx == 0 or sy == 0:
                t += 1
            elif sx == sy:
                c += 1
            else:
                d += 1
    tot = n * (n - 1) // 2
    return (c - d) / tot, c, d, t


def series_stats(doses, Fs, H_by_dose, pi, rng):
    tau, c, d, t = kendall_tau(doses, Fs)
    b1 = boot_F(H_by_dose[doses[-1]], pi, N_BOOT, rng)
    b0 = boot_F(H_by_dose[doses[0]], pi, N_BOOT, rng)
    gain = b1 - b0
    ci = np.percentile(gain, [2.5, 97.5])
    b05 = boot_F(H_by_dose[0.5], pi, N_BOOT, rng)
    return {
        "kendall_tau": tau, "tau_concordant": c, "tau_discordant": d, "tau_tied": t,
        "gain_F1_minus_F0": Fs[-1] - Fs[0],
        "gain_boot_CI95": [float(ci[0]), float(ci[1])],
        "gain_boot_p_le0": float((gain <= 0).mean()),
        "saturation_F1_minus_F05": Fs[-1] - Fs[3],
    }


def main():
    report = ["STAGE F  P3 seed-injection dose-response  "
              f"{ALPH}/{MODE}  (2026-10-24)"]
    t0 = time.time()
    space = D.load_patterns(ALPH, MODE)
    sp_index = {p: j for j, p in enumerate(space)}
    log(report, f"pattern space: {len(space)} patterns")
    pi = {c: load_pi(c) for c in ("CN", "EU")}
    log(report, f"pi_CN sum(norm)=1 ok={abs(pi['CN'].sum()-1)<1e-9}  "
                f"pi_EU ok={abs(pi['EU'].sum()-1)<1e-9}")

    gate = {}
    p1 = RUND / "p1_short23_results.json"
    if p1.exists():
        gate = json.loads(p1.read_text(encoding="utf-8")).get("gate", {})

    seeds = {c: json.loads((RUNF / f"seeds_{c}.json").read_text(encoding="utf-8"))
             for c in ("CN", "EU")}

    results = {"alphabet": ALPH, "mode": MODE, "doses": DOSES, "n_boot": N_BOOT,
               "seed": 20261024, "gate": gate, "series": {}, "cross": {},
               "elapsed_sec": None}

    arms_by_mech = {"M1": ["own"], "M2": ["own", "cross"]}

    for fam in MODELS:
        for cond in ("CN", "EU"):
            H0 = load_stageC_H(fam, cond)
            F0 = 1.0 - D.jsdiv(norm_df(H0.mean(axis=0)), pi[cond])
            log(report, f"\n[{fam} {cond}] stage-C pool n={H0.shape[0]}  F(0)={F0:.4f}")

            for mech, arms in arms_by_mech.items():
                for arm in arms:
                    key = f"{fam}_{mech}_{arm}_{cond}"
                    doses, Fs, ns, covs, H_by = [0.0], [F0], [int(H0.shape[0])], \
                        [float(H0.mean(axis=0).sum())], {0.0: H0}
                    for c in CDOSES:
                        cell = f"{fam}_{mech}_{arm}_{cond}_{c}"
                        H = cell_H(cell, space)
                        df_c = H.mean(axis=0)
                        Fc = 1.0 - D.jsdiv(norm_df(df_c), pi[cond])
                        doses.append(c); Fs.append(Fc); ns.append(int(H.shape[0]))
                        covs.append(float(df_c.sum())); H_by[c] = H
                    st = series_stats(doses, Fs, H_by, pi[cond], RNG)
                    results["series"][key] = {
                        "doses": doses, "F": [float(x) for x in Fs],
                        "n_pieces": ns, "coverage_per_piece": covs, **st}
                    log(report, f"{key:34s} F={[round(x,3) for x in Fs]}  "
                                f"tau={st['kendall_tau']:+.2f}  "
                                f"gain={st['gain_F1_minus_F0']:+.3f} "
                                f"CI95=[{st['gain_boot_CI95'][0]:+.3f},"
                                f"{st['gain_boot_CI95'][1]:+.3f}] "
                                f"p(<=0)={st['gain_boot_p_le0']:.3f}  "
                                f"sat={st['saturation_F1_minus_F05']:+.3f}")
                    if mech == "M2" and arm == "own":
                        for c in CDOSES:
                            bo = boot_F(H_by[c], pi[cond], N_BOOT, RNG)
                            cellx = f"{fam}_M2_cross_{cond}_{c}"
                            Hx = cell_H(cellx, space)
                            bx = boot_F(Hx, pi[cond], N_BOOT, RNG)
                            diff = bo - bx
                            ci = np.percentile(diff, [2.5, 97.5])
                            results["cross"][f"{fam}_{cond}_{c}"] = {
                                "F_own_minus_cross": float(
                                    results["series"][f"{fam}_M2_own_{cond}"]["F"][DOSES.index(c)]
                                    - (1.0 - D.jsdiv(norm_df(Hx.mean(axis=0)), pi[cond]))),
                                "boot_CI95": [float(ci[0]), float(ci[1])],
                                "boot_p_le0": float((diff <= 0).mean())}
                            log(report, f"    cross {cond} c={c}: "
                                f"own-cross={results['cross'][f'{fam}_{cond}_{c}']['F_own_minus_cross']:+.3f} "
                                f"CI95=[{ci[0]:+.3f},{ci[1]:+.3f}] "
                                f"p(<=0)={results['cross'][f'{fam}_{cond}_{c}']['boot_p_le0']:.3f}")

            # ---- M3 output-side filter (control, zero new sampling) ----
            key3 = f"{fam}_M3_filter_{cond}"
            doses3, Fs3, ns3, covs3, H_by3 = [0.0], [F0], [int(H0.shape[0])], \
                [float(H0.mean(axis=0).sum())], {0.0: H0}
            for c in CDOSES:
                sidx = []
                for sd in seeds[cond]["cells"][str(c)]["seeds"]:
                    j = sp_index.get(tuple(sorted(sd)))
                    if j is not None:
                        sidx.append(j)
                sidx = np.array(sorted(set(sidx)), dtype=int)
                hits = H0[:, sidx].sum(axis=1) if len(sidx) else np.zeros(H0.shape[0])
                cov = hits / max(len(sidx), 1)
                K = max(MIN_KEEP, int(np.ceil((1.0 - c) * H0.shape[0])))
                order = sorted(range(H0.shape[0]),
                               key=lambda i: (-cov[i], -hits[i], i))
                keep = np.array(sorted(order[:K]))
                Hk = H0[keep]
                Fk = 1.0 - D.jsdiv(norm_df(Hk.mean(axis=0)), pi[cond])
                doses3.append(c); Fs3.append(Fk); ns3.append(int(Hk.shape[0]))
                covs3.append(float(Hk.mean(axis=0).sum())); H_by3[c] = Hk
                log(report, f"    M3 {cond} c={c}: |S|={len(sidx)} K={K} "
                            f"F={Fk:.3f} (retained min cov={cov[keep].min():.3f})")
            st3 = series_stats(doses3, Fs3, H_by3, pi[cond], RNG)
            results["series"][key3] = {
                "doses": doses3, "F": [float(x) for x in Fs3],
                "n_pieces": ns3, "coverage_per_piece": covs3, **st3}
            log(report, f"{key3:34s} F={[round(x,3) for x in Fs3]}  "
                        f"tau={st3['kendall_tau']:+.2f}  "
                        f"gain={st3['gain_F1_minus_F0']:+.3f} "
                        f"CI95=[{st3['gain_boot_CI95'][0]:+.3f},"
                        f"{st3['gain_boot_CI95'][1]:+.3f}] "
                        f"p(<=0)={st3['gain_boot_p_le0']:.3f}  "
                        f"sat={st3['saturation_F1_minus_F05']:+.3f}")
            results["elapsed_sec"] = round(time.time() - t0, 1)
            save_results(results, report)
            log(report, f"[checkpoint] {fam}/{cond} saved")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    save_results(results, report)
    log(report, f"\n[saved] {RES_JSON}  elapsed {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()
