# Repo copy: paths resolved relative to repository root; regenerate runs/d caches first.
# -*- coding: utf-8 -*-
"""Analysis A (reviewer D): full CN x EU cell-pair distance matrix for P1.
Point estimates and 95% bootstrap CIs for D_JS between every Chinese-prompted
cell and every European-prompted cell at the decisive a3 encoding (mode short23),
plus point estimates at a1/a2. Replication: q = df/df.sum(), jsdiv as in stage_d."""
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[3]   # repository root
RUNS = WORK / "runs"
MODE = "short23"
MODELS = {"deepseek": "A", "zhipu": "B", "dashscope": "C"}
N_BOOT = 1000
SEED = 20261023


def js_rows(P, Q):
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(P > 0, P * np.log2(P / M), 0.0).sum(axis=1)
        t2 = np.where(Q > 0, Q * np.log2(Q / M), 0.0).sum(axis=1)
    return 0.5 * (t1 + t2)


def norm_rows(C):
    s = C.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return C / s


def multinomial_counts(n_clus, n_boot, rng):
    idx = rng.integers(0, n_clus, size=(n_boot, n_clus), dtype=np.int64)
    flat = (np.arange(n_boot, dtype=np.int64)[:, None] * n_clus + idx).ravel()
    C = np.bincount(flat, minlength=n_boot * n_clus).astype(np.float32)
    return C.reshape(n_boot, n_clus)


def boot_gap_rows(H1, H2, n_boot, rng):
    gaps = []
    for H in (H1, H2):
        n = H.shape[0]
        C = multinomial_counts(n, n_boot, rng)
        d = norm_rows(C @ H.astype(np.float32))
        gaps.append(d)
    return js_rows(gaps[0], gaps[1])


def qvec(name, alph):
    df = np.load(RUNS / "d/df" / f"out_{name}.{alph}.{MODE}.npy")
    return df / df.sum()


def hits(name, alph):
    return np.load(RUNS / "d/hcache" / f"out_{name}.{alph}.{MODE}.npy")


result = {}
for alph in ("a1", "a2", "a3"):
    cells_cn = {m: qvec(f"{k}_CN", alph) for k, m in MODELS.items()}
    cells_eu = {m: qvec(f"{k}_EU", alph) for k, m in MODELS.items()}
    mat = {}
    for mc, qc in cells_cn.items():
        for me, qe in cells_eu.items():
            mat[f"{mc}CN-{me}EU"] = float(js_rows(qc[None, :], qe[None, :])[0])
    result[alph] = mat

# bootstrap CIs at a3
rng = np.random.default_rng(SEED)
ci = {}
for mc in MODELS.values():
    for me in MODELS.values():
        H1 = hits([k for k, v in MODELS.items() if v == mc][0] + "_CN", "a3")
        H2 = hits([k for k, v in MODELS.items() if v == me][0] + "_EU", "a3")
        b = boot_gap_rows(H1, H2, N_BOOT, rng)
        ci[f"{mc}CN-{me}EU"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
        print(f"a3 {mc}CN-{me}EU  D={result['a3'][f'{mc}CN-{me}EU']:.3f}  CI95=[{ci[f'{mc}CN-{me}EU'][0]:.3f}, {ci[f'{mc}CN-{me}EU'][1]:.3f}]")

out = {"mode": MODE, "n_boot": N_BOOT, "seed": SEED,
       "point": result, "ci95_a3": ci,
       "ref_a3": {"human_gap": 0.978, "N2_scale_matched": 0.982,
                  "note": "human gap and N2 from p1_short23_results.json a3 (rounded)"}}
json.dump(out, open(WORK / "runs/analysis_A_cellpairs.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print()
print("a1 matrix:"); [print(" ", k, round(v, 3)) for k, v in result["a1"].items()]
print("a2 matrix:"); [print(" ", k, round(v, 3)) for k, v in result["a2"].items()]
print("saved _build/analysis_A_cellpairs.json")
