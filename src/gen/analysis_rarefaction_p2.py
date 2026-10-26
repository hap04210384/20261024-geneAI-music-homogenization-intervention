# Repo copy: paths resolved relative to repository root; regenerate runs/d caches first.
# -*- coding: utf-8 -*-
"""Analysis B (reviewer D): per-cell corpus-size accounting + rarefaction control
for P2 tail extinction. Shows (i) per-cell pieces/windows/distinct patterns and
(ii) that tail extinction is not a mechanical sample-size effect: the human
reference, subsampled to the same number of windows as each output cell, retains
far higher tail survival than the model output."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[3]   # repository root
RUNS = WORK / "runs"
MODE = "short23"
ALPH = "a3"
MODELS = {"deepseek": "A", "zhipu": "B", "dashscope": "C"}
N_REP = 500
SEED = 20261026

# ---------- frozen strata (exactly as stage_e) ----------
fr = json.loads((RUNS / "e" / f"strata_{ALPH}.json").read_text(encoding="utf-8"))
strata = {k: np.asarray(v, dtype=np.int64) for k, v in fr["S"].items()}
df_cn = np.load(RUNS / "d/df" / f"human_pop909.{ALPH}.{MODE}.npy")   # pi_CN
df_eu = np.load(RUNS / "d/df" / f"human_mtc.{ALPH}.{MODE}.npy")      # pi_EU
n_cn, n_eu = 908, 17638
mass = df_cn * n_cn + df_eu * n_eu
pi = {"CN": df_cn, "EU": df_eu}


def sigma(df_q, ref, cols_all, mass):
    out = {}
    for k in ("S1", "S2", "S3"):
        cols = cols_all[k][ref[cols_all[k]] > 0]
        m = mass[cols] / mass[cols].sum()
        out[k] = float((m * (df_q[cols] / ref[cols])).sum())
    return out


rng = np.random.default_rng(SEED)
report = {"alphabet": ALPH, "mode": MODE, "n_rep": N_REP, "seed": SEED,
          "human": {}, "cells": {}}

# human references
for cond, (df_h, n_h) in (("CN", (df_cn, n_cn)), ("EU", (df_eu, n_eu))):
    W = float(df_h.sum() * n_h)
    report["human"][cond] = {
        "n_pieces": n_h, "n_windows": int(round(W)),
        "windows_per_piece": W / n_h,
        "distinct_patterns": int((df_h > 0).sum()),
        "sigma_full": sigma(df_h, df_h, strata, mass),
    }

# output cells
for key, fam in MODELS.items():
    for cond in ("CN", "EU"):
        H = np.load(RUNS / "d/hcache" / f"out_{key}_{cond}.{ALPH}.{MODE}.npy")
        n = H.shape[0]
        W = int(H.sum())
        df_q = H.mean(axis=0)
        ref = pi[cond]
        sig_obs = sigma(df_q, ref, strata, mass)
        # rarefaction: human reference at the cell's own window count
        p_ref = ref / ref.sum()
        rare = {k: np.empty(N_REP) for k in ("S1", "S2", "S3")}
        for r in range(N_REP):
            cnt = rng.multinomial(W, p_ref)
            # per-piece mean over the cell's own piece count -> same units as df_q
            df_r = cnt / n
            s = sigma(df_r, ref, strata, mass)
            for k in rare:
                rare[k][r] = s[k]
        cell = {
            "family": fam, "condition": cond, "n_pieces": n,
            "n_windows": W, "windows_per_piece": W / n,
            "distinct_patterns": int((df_q > 0).sum()),
            "sigma_observed": sig_obs,
            "human_at_same_W": {k: [float(np.mean(rare[k])), float(np.percentile(rare[k], 2.5)),
                                    float(np.percentile(rare[k], 97.5))] for k in rare},
        }
        report["cells"][f"{fam}_{cond}"] = cell
        print(f"{fam}_{cond}: pieces={n} windows={W} win/piece={W/n:.1f} "
              f"distinct={cell['distinct_patterns']}")
        for k in ("S1", "S2", "S3"):
            mu, lo, hi = cell["human_at_same_W"][k]
            print(f"   {k}: obs={sig_obs[k]:.3f}  human@W={mu:.3f} [{lo:.3f},{hi:.3f}]")

json.dump(report, open(WORK / "runs/analysis_B_rarefaction.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("saved _build/analysis_B_rarefaction.json")
