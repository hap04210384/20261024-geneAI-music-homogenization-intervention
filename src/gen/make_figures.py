# -*- coding: utf-8 -*-
"""Generate manuscript figures F1, F2, F3, F5 as SVG, then EMF via Inkscape.
Data sources: runs/d/p1_short23_results.json, runs/e/p2_results.json,
runs/f/p3_results.json. Output: Manuscript_work/figures/F{n}.svg/.emf
"""
import gzip
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Ellipse, PathPatch)
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties

WORK = Path(__file__).resolve().parents[3]          # Manuscript_work
RUNS = WORK / "runs"
FIGDIR = WORK / "figures"
FIGDIR.mkdir(exist_ok=True)
INKSCAPE = r"C:/Program Files/Inkscape/bin/inkscape.exe"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "svg.fonttype": "path",   # keep text as paths so EMF has no font issues
})

MODELS = [("deepseek", "Model A"), ("zhipu", "Model B"), ("dashscope", "Model C")]
ENCS = [("a1", "a1"), ("a2", "a2"), ("a3", "a3")]

# unified palette: deep navy -> vivid mid blue -> light sky, warm red accent
C_DARK, C_MID, C_LIGHT = "#154360", "#2e86c1", "#85c1e9"
C_RED, C_GRAY = "#c0392b", "#707b7c"
C_SHADE, C_AMBER = "#fdebd0", "#9c4a08"

# red-orange-blue palette for F2-F5 (consistent semantics across figures):
# deep blue = primary emphasis / a3 / own-culture arm; vivid blue = main series;
# azure/sky = lighter encodings / cross-culture arm; orange = secondary category
# (a2 / Model B / M1 priming); red = human reference / control / contrast
V_DEEP, V_BLUE, V_AZURE = "#0D47A1", "#1565C0", "#1E88E5"
V_SKY = "#64B5F6"
V_ORANGE, V_RED = "#EF6C00", "#E53935"
V_GREEN = "#43A047"
V_DORANGE = "#E65100"
V_REFGRAY = "#424242"

p1 = json.load(open(RUNS / "d/p1_short23_results.json", encoding="utf-8"))
p2 = json.load(open(RUNS / "e/p2_results.json", encoding="utf-8"))
p3 = json.load(open(RUNS / "f/p3_results.json", encoding="utf-8"))


def save(fig, name):
    svg = FIGDIR / f"{name}.svg"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    emf = FIGDIR / f"{name}.emf"
    r = subprocess.run([INKSCAPE, str(svg), "--export-type=emf",
                        f"--export-filename={emf}"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and emf.exists(), (name, r.stderr[-500:])
    print("OK", emf)



# ---------------- F1: study-design teaser with notated example pair ----------------
# Layout and palette are intentionally sans-serif; F2-F5 restore the serif rcParams.
_rc_saved = plt.rcParams.copy()
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",
    "axes.edgecolor": "#9aa5b1",
    "axes.linewidth": 0.7,
    "font.size": 10,
})

SYM = FontProperties(family="Segoe UI Symbol")

NAVY   = "#1f3a5f"
TEAL   = "#0e8a7d"
ORANGE = "#e8862e"
RED    = "#c0392b"
GREY   = "#6b7684"
LGREY  = "#e7eaee"
INK    = "#222a33"

fig = plt.figure(figsize=(13.2, 7.92))
fig.patch.set_facecolor("white")

# ------------------------------------------------- symmetric panel layout ----
PX0, PW, PGAP = 0.012, 0.2275, 0.022
XS = [PX0 + i * (PW + PGAP) for i in range(4)]
PXEND = XS[-1] + PW
HDY0, HDY1 = 0.915, 0.99
YB, USTUB = 0.145, 0.075
YLOW = 0.175

panels = [
    dict(letter="a", title="SETUP —\none shared channel",   color=NAVY,   tint="#edf1f7"),
    dict(letter="b", title="P1 —\ngap amplification",         color=TEAL,   tint="#e6f3f1"),
    dict(letter="c", title="P2 —\nstructured collapse",       color=ORANGE, tint="#fdf1e3"),
    dict(letter="d", title="P3 —\nzero-training repair",      color=RED,    tint="#fbecea"),
]

for p, x0 in zip(panels, XS):
    fig.patches.append(Rectangle((x0, HDY0), PW, HDY1 - HDY0,
                                 fc=p["color"], ec="none",
                                 transform=fig.transFigure, zorder=2))
    fig.text(x0 + 0.0175, (HDY0 + HDY1) / 2, p["letter"],
             fontsize=16, fontweight="bold", color=p["color"],
             ha="center", va="center", transform=fig.transFigure, zorder=3,
             bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.018", fc="white", ec="none"))
    fig.text(x0 + 0.048, (HDY0 + HDY1) / 2, p["title"],
             fontsize=14, fontweight="bold", color="white",
             ha="left", va="center", transform=fig.transFigure, zorder=3)
    fig.lines.append(Line2D([x0, x0, x0 + PW, x0 + PW],
                            [YB + USTUB, YB, YB, YB + USTUB],
                            color="#c7cdd4", lw=1.8,
                            solid_capstyle="round", solid_joinstyle="round",
                            transform=fig.transFigure, zorder=1))

for gc in (XS[0] + PW + PGAP / 2, XS[1] + PW + PGAP / 2, XS[2] + PW + PGAP / 2):
    fig.patches.append(FancyArrowPatch((gc - 0.0065, 0.55), (gc + 0.0065, 0.55),
                                       arrowstyle="-|>", mutation_scale=17, lw=2.4,
                                       color="#8b95a1", transform=fig.transFigure,
                                       zorder=4))

def mini_ax(x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor("white")
    ax.set_zorder(3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9.5, length=2.5)
    return ax

CA, CB, CC, CD = [x + PW / 2 for x in XS]

# ============================================================ panel a ========
ax = mini_ax(XS[0] + 0.014, 0.72, PW - 0.028, 0.145)
models = ["DeepSeek", "Qwen", "GLM"]
cn = [799, 232, 867]
eu = [941, 505, 962]
xp = np.arange(3)
b1 = ax.bar(xp - 0.19, cn, 0.36, color=NAVY, label="CN prompt")
b2 = ax.bar(xp + 0.19, eu, 0.36, color="#7fa3cc", label="EU prompt")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{int(r.get_height())}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(xp); ax.set_xticklabels(models)
ax.set_ylim(0, 2200); ax.set_yticks([])
ax.set_ylabel("melodies", fontsize=11)
ax.legend(fontsize=10, frameon=False, loc="upper right", handlelength=1.0)
ax.set_title("4,306 melodies — 3 families × 2 prompts",
             fontsize=11, color=INK, pad=4)

ax = mini_ax(XS[0] + 0.042, 0.435, PW - 0.084, 0.15)
vals = [18109, 909]
ax.barh([1, 0], vals, 0.55, color=[TEAL, ORANGE])
ax.text(400, 1, "MTC-FS-INST · EU folk", ha="left", va="center",
        fontsize=9.5, color="white", fontweight="bold")
ax.text(18109 + 1500, 1, "18,109 (raw)", va="center", fontsize=10.5, color=INK)
ax.text(909 + 400, 0, "POP909 · 909 Chinese pop songs (raw)", va="center", fontsize=10.5, color=INK)
ax.set_xlim(0, 23800); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Human references (frozen, pre-AI) —\nculture signal lives in the libraries",
             fontsize=11, color=INK, pad=4, fontweight="bold")

fig.text(CA, 0.407, "one generic output space Q —\nsame models, same prompts\nper culture, zero parameter updates",
         fontsize=11.5, color=GREY, ha="center", va="center", style="italic")
fig.text(CA, 0.322, "gap signature:\nsurface style survives,\nrecognition core does not",
         fontsize=12.5, fontweight="bold", color=NAVY, ha="center", va="center")
fig.text(CA, YLOW, "encode (a1–a3) → frozen libraries\n→ 3 nulls N0–N2 → repairs M1–M3",
         fontsize=11, color=GREY, ha="center", va="bottom")

# ============================================================ panel b ========
ax = mini_ax(XS[1] + 0.014, 0.605, PW - 0.028, 0.265)
labels = ["Human", "Qwen", "GLM", "DeepSeek"]
gaps = [0.357, 0.447, 0.864, 1.000]
mult = ["", "×1.3", "×2.4", "×2.8"]
cols = ["#4d4d4d", "#7fa3cc", "#9b7fd4", "#e08585"]
ax.bar(range(4), gaps, 0.62, color=cols)
for i, (g, m) in enumerate(zip(gaps, mult)):
    if m:
        ax.text(i, g + 0.025, m, ha="center", fontsize=11,
                fontweight="bold", color=TEAL)
ax.axhline(0.357, ls="--", lw=1.0, color=GREY)
ax.text(-0.44, 0.385, "human gap", fontsize=9.5, color=GREY, ha="left")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0])
ax.set_xlim(-0.55, 3.55)
ax.set_ylabel("distributional gap  $D_{JS}$", fontsize=11)
ax.set_title("gap amplification vs. human cultures (a2)", fontsize=11, pad=4)

fig.text(CB, 0.515, "gap(Q) up to 2.8× the human gap",
         fontsize=13, fontweight="bold", color=TEAL, ha="center")
ax = mini_ax(XS[1] + 0.042, 0.27, PW - 0.070, 0.215)
dv = [0.973, 0.987, 0.653, 0.789]
cols2 = [TEAL, TEAL, "#e08585", "#e08585"]
ax.barh(range(4), dv, 0.6, color=cols2)
for i, v in enumerate(dv):
    ax.text(v - 0.02, i, f"{v:.2f}", va="center", ha="right", fontsize=10, color="white")
ax.set_yticks(range(4))
ax.set_yticklabels(["$Q_{CN}$→$π_{CN}$", "$Q_{CN}$→$π_{EU}$",
                    "$Q_{EU}$→$π_{EU}$", "$Q_{EU}$→$π_{CN}$"], fontsize=9.5)
ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1.0])
ax.invert_yaxis()
ax.set_title("outputs leave their own reference", fontsize=11, pad=4)
fig.text(CB, YLOW, "N1 prompt-shuffle: p < 0.001\nnot sampling noise (n = 1000)",
         fontsize=11, color=GREY, ha="center", va="bottom")

# ============================================================ panel c ========
ax = mini_ax(XS[2] + 0.014, 0.65, 0.102, 0.22)
labels = ["human π", "DeepSeek", "GLM", "Qwen"]
H  = [6.77, 4.86, 5.56, 6.44]
drop = ["", "−28%", "−18%", "−5%"]
cols3 = ["#4d4d4d", ORANGE, "#f2b05c", "#f5cf9a"]
ax.bar(range(4), H, 0.62, color=cols3)
for i, (h, d) in enumerate(zip(H, drop)):
    ax.text(i, h + 0.12, d if d else "6.77 bit", ha="center", fontsize=9.5,
            fontweight="bold", color=RED if d else GREY)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9, rotation=22, ha="right")
ax.set_ylim(0, 8.2); ax.set_yticks([0, 4, 8])
ax.set_ylabel("entropy H (bit)", fontsize=10)
ax.set_title("entropy shrinks (a1)", fontsize=11, pad=3)

ax = mini_ax(XS[2] + 0.126, 0.65, 0.102, 0.22)
S1 = [0.90, 1.16, 2.72]; S2 = [0.00, 0.01, 0.19]; S3 = [0.00, 0.00, 0.00]
xp = np.arange(3)
ax.bar(xp - 0.26, S1, 0.24, color=ORANGE, label="S1 core")
ax.bar(xp,        S2, 0.24, color="#f2b05c", label="S2 tail")
ax.bar(xp + 0.26, S3, 0.24, color="#f5cf9a", label="S3 tip")
ax.set_xticks(xp); ax.set_xticklabels(["DeepSeek", "GLM", "Qwen"], fontsize=9)
ax.set_ylim(0, 3.1); ax.set_yticks([0, 1, 2, 3])
ax.set_ylabel("mass shift σ", fontsize=10)
ax.legend(fontsize=9, frameon=False, loc="upper left", handlelength=1.0)
ax.set_title("strata collapse", fontsize=11, pad=3)

fig.text(CC, 0.575, "the tail dies first —\nnon-core strata\ndie wholesale",
         fontsize=12, fontweight="bold", color=RED, ha="center", va="center")
fig.text(CC, 0.505, "strata: S1 core · S2 tail · S3 tip\n(frozen frequency strata, Suppl. §S3)\nrobust across alphabets a1–a3 & windows",
         fontsize=11, color=GREY, ha="center", va="center")
fig.text(CC, 0.40, "entropy deficit is structural:\nthe rare-pattern reservoir\ndisappears",
         fontsize=11.5, color=INK, ha="center", va="center", style="italic")
fig.text(CC, 0.325, "entropy deficit at every encoding:\np < 0.001 in all 18 cells (Table 3)",
         fontsize=11, color=GREY, ha="center", va="center")
fig.text(CC, YLOW, "diagnosis precedes repair —\nall claims bound to\ncounted patterns",
         fontsize=12, fontweight="bold", color=ORANGE, ha="center", va="bottom")

# ============================================================ panel d ========
ax = mini_ax(XS[3] + 0.014, 0.545, PW - 0.028, 0.325)
doses = [0, 0.1, 0.25, 0.5, 1.0]
m2_ds = [0.140, 0.095, 0.094, 0.265, 0.418]
m2_qw = [0.128, 0.094, 0.126, 0.213, 0.454]
m2_gl = [0.107, 0.062, 0.249, 0.343, 0.293]
m3_qw = [0.128, 0.128, 0.123, 0.117, 0.107]
ax.plot(doses, m2_ds, "-o", ms=4, lw=2.0, color=NAVY,  label="M2 structure-seed · DeepSeek")
ax.plot(doses, m2_qw, "-s", ms=4, lw=2.0, color=RED,   label="M2 structure-seed · Qwen")
ax.plot(doses, m2_gl, "-^", ms=4, lw=2.0, color=TEAL,  label="M2 structure-seed · GLM")
ax.plot(doses, m3_qw, "--v", ms=3.6, lw=1.6, color=GREY, label="M3 naive-filter · Qwen")
ax.set_xlabel("seed dose  c", fontsize=11)
ax.set_ylabel("pattern-faithfulness  F", fontsize=11)
ax.set_ylim(0, 0.60); ax.set_xticks(doses)
ax.legend(fontsize=10, frameon=False, loc="upper left", handlelength=1.4)
ax.set_title("repair vs. filter degradation (CN)", fontsize=11, pad=4)

fig.text(CD, 0.44, "gain up to +0.33\n(Qwen·CN, c = 1.0)",
         fontsize=18, fontweight="bold", color=RED, ha="center", va="center")
fig.text(CD, 0.35, "Kendall τ monotone up (p < 0.01)\nzero parameter updates —\nno fine-tuning, no RLHF, no prompts",
         fontsize=11.5, color=INK, ha="center", va="center")
fig.text(CD, 0.263, "M1 (ad-hoc injection) gains little;\nM3 (filtering) hurts —\nstructure must be re-seeded, not re-selected",
         fontsize=10.5, color=GREY, ha="center", va="center")
fig.text(CD, YLOW, "every +ΔF is cited to a counted\npattern cohort (Sec. 5, Table 5)",
         fontsize=11, color=GREY, ha="center", va="bottom")

# ============================================================ footer =========
FT_Y0, FT_H = 0.016, 0.128
fig.patches.append(Rectangle((PX0, FT_Y0), PXEND - PX0, FT_H, fc=LGREY,
                             ec="none", transform=fig.transFigure, zorder=1))

# ---------------- left zone: notated human-vs-model example ----------------
EX0, EXW = PX0 + 0.008, 0.190          # example axes rect
axm = fig.add_axes([EX0, 0.026, EXW, 0.118])
axm.set_facecolor("none")
axm.set_xlim(0, 34); axm.set_ylim(-1.1, 15.2)
axm.set_zorder(3)
axm.axis("off")
for _s in axm.spines.values():
    _s.set_visible(False)

STEP_DIA = {0: 0, 2: 1, 4: 2, 5: 3, 6: 3, 7: 4, 9: 5, 11: 6}   # chroma -> diatonic step
def dia(m):
    return (m // 12 - 1) * 7 + STEP_DIA[m % 12]

def load_human():
    with gzip.open(WORK / "data_backup/mtc/MTC-FS-INST-2.0_sequences-1.1.jsonl.gz",
                   "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["id"] == "NLB125823_01":
                break
    F = rec["features"]
    notes = []
    for i in range(len(F["midipitch"])):
        if F["onsettick"][i] >= 24:      # first 2 bars (12 ticks per bar)
            break
        notes.append((F["midipitch"][i], F["duration"][i] * 2))   # dur in eighths
    return notes

def load_model():
    d = json.load(open(WORK / "runs/c/gate_deepseek_EU_20261023-111519.json",
                       encoding="utf-8"))
    abc = [x["text"] for x in d["replies"] if x.get("i") == 7][0]
    body = abc.split("K:G", 1)[1].replace("|]", "|")
    STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    seq, total = [], 0.0
    for m in re.finditer(r"([A-Ga-g])([\',]*)(\d*)(/\d+)?", body):
        letter, acc, num, frac = m.groups()
        midi = STEP[letter.upper()] + (60 if letter.isupper() else 72)
        for a in acc:
            midi += 12 if a == "'" else -12
        if letter.upper() == "F":
            midi += 1                                   # key of G: F sharp
        dur = int(num) if num else 1
        if frac:
            dur = dur / int(frac[1:])
        if total + dur > 12:                            # first 2 bars only
            break
        seq.append((midi, float(dur)))
        total += dur
    return seq

def draw_staff(ax, notes, yb, x0, label, label_color, front_w=5.4, ew=0.72,
               bar_e=6, total_e=12):
    W = front_w + total_e * ew + 1.2
    for i in range(5):
        ax.plot([x0, x0 + W], [yb + i] * 2, color="#39424e", lw=0.9, zorder=2)
    ax.text(x0 + 1.05, yb + 2.0, "\U0001D11E", fontproperties=SYM,
            fontsize=17, ha="center", va="center", color="#39424e")
    sx = x0 + 2.10
    for dx in (0, 0.26):
        ax.plot([sx + dx] * 2, [yb + 3.35, yb + 4.65], color="#39424e", lw=1.1)
    for dy in (3.85, 4.35):
        ax.plot([sx - 0.10, sx + 0.36], [yb + dy - 0.13, yb + dy + 0.13],
                color="#39424e", lw=1.1)
    ax.text(x0 + 3.15, yb + 2.95, "6", fontsize=11, ha="center", va="center",
            fontweight="bold", color="#39424e")
    ax.text(x0 + 3.15, yb + 1.05, "8", fontsize=11, ha="center", va="center",
            fontweight="bold", color="#39424e")
    ax.text(x0, yb + 5.35, label, fontsize=7.5, ha="left", va="center",
            fontweight="bold", color=label_color)

    n = len(notes)
    ds = [d for _, d in notes]
    starts = [0.0] * n
    for k in range(1, n):
        starts[k] = starts[k - 1] + ds[k - 1]
    xs = [x0 + front_w + (st + d / 2) * ew for (st, d) in zip(starts, ds)]
    ys = [yb + (dia(m) - 30) / 2 for m, _ in notes]

    for m in range(1, total_e // bar_e):
        bx = x0 + front_w + m * bar_e * ew
        ax.plot([bx, bx], [yb - 0.12, yb + 4.12], color="#39424e", lw=0.9, zorder=2)
    xend = x0 + front_w + total_e * ew
    ax.plot([xend, xend], [yb - 0.12, yb + 4.12], color="#39424e", lw=0.9)
    ax.plot([xend + 0.20, xend + 0.20], [yb - 0.12, yb + 4.12], color="#39424e", lw=2.6)

    stem_up = [ys[k] < yb + 2 for k in range(n)]
    i = 0
    while i < n:
        if ds[i] <= 1.5:
            j = i
            while (j + 1 < n and ds[j + 1] <= 1.5
                   and (starts[j + 1] % bar_e) > 1e-9):
                j += 1
            run = list(range(i, j + 1))
            if len(run) >= 2:
                up = bool(np.mean([ys[k] for k in run]) < yb + 2)
                for k in run:
                    stem_up[k] = up
                tips = [(xs[k], ys[k] + (3.0 if up else -3.0)) for k in run]
                ax.plot([tips[0][0], tips[-1][0]], [tips[0][1], tips[-1][1]],
                        color="#222a33", lw=2.6, solid_capstyle="butt", zorder=4)
                for k in run:
                    if ds[k] <= 0.75:
                        ax.plot([xs[k], xs[k] - 0.80],
                                [ys[k] + (2.65 if up else -2.65),
                                 ys[k] + (2.15 if up else -2.15)],
                                color="#222a33", lw=2.0, zorder=4)
            i = j + 1
        else:
            i += 1

    for k in range(n):
        x, y, d = xs[k], ys[k], ds[k]
        up = stem_up[k]
        xstem = x + (0.20 if up else -0.20)
        ax.plot([xstem, xstem], [y + (0.26 if up else -0.26), y + (3.0 if up else -3.0)],
                color="#222a33", lw=1.0, zorder=3)
        if d <= 1.5:
            nb = (ds[k - 1] <= 1.5) if k > 0 else False
            na = (ds[k + 1] <= 1.5) if k + 1 < n else False
            if not (nb or na):
                fy = y + (3.0 if up else -3.0)
                sgn = 1 if up else -1
                fdir = 1 if up else -1
                verts = [(xstem, fy),
                         (xstem + fdir * 0.70, fy + sgn * 0.52),
                         (xstem + fdir * 1.05, fy + sgn * 0.12),
                         (xstem + fdir * 0.50, fy - sgn * 0.26)]
                ax.add_patch(PathPatch(MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                                                    MplPath.CURVE4, MplPath.CURVE4]),
                                       fill=False, edgecolor="#222a33", lw=1.3, zorder=4))
        if abs(d - 1.5) < 1e-6 or abs(d - 3) < 1e-6:
            ax.plot([x + 0.58], [y], marker="o", ms=1.8, color="#222a33", zorder=4)
        ax.add_patch(Ellipse((x, y), width=0.80, height=0.52, angle=-16,
                             fc="#222a33", ec="none", zorder=5))
    return W

hn = load_human()
mn = load_model()
w1 = draw_staff(axm, hn, yb=7.6, x0=0.10,
                label="human · MTC, 1884",
                label_color=NAVY)
w2 = draw_staff(axm, mn, yb=7.6, x0=17.0,
                label="Model A · DeepSeek",
                label_color=ORANGE)
axm.text(0.10, 5.50, "human tune “Schep vreugde in het leven”:",
         fontsize=7.2, color=INK, ha="left", va="center")
axm.text(0.10, 3.65, "same metre, key, and dance pulse —", fontsize=7.2, style="italic",
         color=GREY, ha="left", va="center")
axm.text(0.10, 1.80, "0 of the tune’s 47 four-note patterns", fontsize=7.2, style="italic",
         color=RED, ha="left", va="center", fontweight="bold")
axm.text(0.10, 0.10, "appear in the output", fontsize=7.2, style="italic",
         color=RED, ha="left", va="center", fontweight="bold")

# ---------------- right zone: four quantitative claims ----------------
claims = [
    (0.030, "4,306", "generated melodies audited —\n3 families, 1 protocol, no tuning"),
    (0.283, "2.8×", "max gap ratio vs. humans (a2);\nΔ$_D$ +0.70 at a3, p < 0.001"),
    (0.536, "−28%", "max entropy deficit (a1·CN);\nnon-core strata σ ≈ 0, extinct"),
    (0.789, "+0.33", "max repair gain by seeding,\ndose-dependent, zero-training"),
]
footer_texts = []
for _x, big, small in claims:
    t_big = fig.text(0, 0.076, big, fontsize=14.0, fontweight="bold", color=INK,
                     ha="center", va="center", transform=fig.transFigure, zorder=3)
    t_small = fig.text(0, 0.076, small, fontsize=8.0, color="#3d4754",
                       ha="left", va="center", transform=fig.transFigure, zorder=3)
    footer_texts.append((t_big, t_small))

fig.canvas.draw()
_rdr = fig.canvas.get_renderer()
_BW = fig.bbox.width
GAP_IN = 0.006
_m = []
for t_big, t_small in footer_texts:
    wb = t_big.get_window_extent(renderer=_rdr).width / _BW
    ws = t_small.get_window_extent(renderer=_rdr).width / _BW
    _m.append((wb, ws))

_start = EX0 + EXW + 0.018
_span = (PXEND - 0.010) - _start
_total = sum(wb + GAP_IN + ws for wb, ws in _m)
_gap = (_span - _total) / 3
_cur = _start
for (t_big, t_small), (wb, ws) in zip(footer_texts, _m):
    t_big.set_x(_cur + wb / 2)
    t_small.set_x(_cur + wb + GAP_IN)
    _cur += wb + GAP_IN + ws + _gap
print("footer gap between groups: %.4f" % _gap)

_f1_svg = FIGDIR / "F1.svg"
fig.savefig(_f1_svg, format="svg", pad_inches=0)
plt.close(fig)
_f1_emf = FIGDIR / "F1.emf"
_r = subprocess.run([INKSCAPE, str(_f1_svg), "--export-type=emf",
                     f"--export-filename={_f1_emf}"],
                    capture_output=True, text=True)
assert _r.returncode == 0 and _f1_emf.exists(), _r.stderr[-500:]
plt.rcParams.update(_rc_saved)
print("OK", _f1_emf)




# ---------------- F2: Delta_D forest plot ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
rows = []
for enc, _ in ENCS:
    for key, label in MODELS:
        v = p1["alphabets"][enc]["models"][key]
        rows.append((enc, label, v["Delta_D"], *v["boot_Delta_CI95"],
                     v["N2_scale_matched_human_gap"]["mean"]))
n = len(rows)
ypos = list(range(n - 1, -1, -1))
enc_colors = {"a1": V_SKY, "a2": V_ORANGE, "a3": V_DEEP}
for (enc, label, d, lo, hi, ref), y in zip(rows, ypos):
    c = enc_colors[enc]
    ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(d, y, "o", color=c, ms=5.5)
# compression zone shading (Delta_D > 0) + zero line
ax.axvspan(0, 1.38, color=C_SHADE, zorder=0)
ax.text(0.40, n - 0.7, "compression zone ($\\Delta_D > 0$): outputs converge\ncloser to each other than same-culture human music does",
        fontsize=6.6, color=V_DORANGE, va="top", ha="left", linespacing=1.3)
ax.axvline(0, color="#666", lw=0.8)
# group separators + human-gap reference lines, group labels, per-point values
bounds = {0: n, 1: n - 3, 2: n - 6, 3: 0}
for gi in range(3):
    ytop = n - 3 * gi - 0.5; ybot = ytop - 3
    ref = rows[gi * 3][5]
    ax.axvline(ref, ymin=(ybot + 0.5) / (n + 1), ymax=(ytop + 0.5) / (n + 1),
               color=V_RED, ls=(0, (4, 3)), lw=1.2)
    ax.text(ref, ytop - 0.45, f"human gap {ref:.3f}", fontsize=6.6,
            color=V_RED, ha="center", va="top", fontweight="bold")
    ax.text(1.335, (ytop + ybot) / 2, f"a{gi + 1}", fontsize=10,
            fontweight="bold", color=V_DEEP, ha="left", va="center")
for (enc, label, d, lo, hi, ref), y in zip(rows, ypos):
    ax.text(hi + 0.02, y, f"{d:+.3f}", fontsize=6.6, color=enc_colors[enc],
            va="center", ha="left", fontweight="bold")
for i in (3, 6):
    ax.axhline(i - 0.5, color="#bbb", lw=0.6, ls=":")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{lab} \u00b7 {enc}" for enc, lab, *_ in rows], fontsize=8)
ax.set_xlabel("Cultural-gap compression  $\\Delta_D$  (95% bootstrap CI; point labels show $\\Delta_D$)", fontsize=9)
ax.set_xlim(-0.45, 1.38)
ax.set_ylim(-0.6, n + 0.6)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
save(fig, "F2")


# ---------------- F3: survival profiles sigma(S_k) ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
strata = ["S1 (head)", "S2 (middle)", "S3 (tail)"]
mk = {"Model A": ("o", V_BLUE), "Model B": ("s", V_ORANGE), "Model C": ("^", V_RED)}
off = {"Model A": 1.35, "Model B": 1.0, "Model C": 0.72}   # vertical label offsets
for i, (ax, (cult, title)) in enumerate(zip(axes, [("CN", "Chinese-prompted output"),
                                                   ("EU", "European-prompted output")])):
    for key, label in MODELS:
        vals = [p2["alphabets"]["a3"]["models"][key][cult]["T2_survival"][f"S{k}"]
                for k in (1, 2, 3)]
        m, c = mk[label]
        ax.plot(range(3), vals, marker=m, ms=5, lw=1.5, color=c, label=label)
        for k, v in enumerate(vals):
            ax.text(k + 0.09, v * off[label], f"{v:.2g}", fontsize=5.6,
                    color=c, va="center", ha="left", fontweight="bold")
    ax.axhline(1.0, color=V_REFGRAY, ls=(0, (4, 3)), lw=1.2)
    ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels(strata, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.3, 2.3)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    ax.text(0.03, 0.97, f"({chr(65 + i)})", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=V_DEEP, ha="left", va="top")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("Survival rate $\\sigma(S_k)$  (log scale)", fontsize=9)
axes[0].text(0.03, 0.88, "non-core strata collapse below 0.1",
             fontsize=6.6, color=V_DEEP, ha="left", va="top",
             transform=axes[0].transAxes)
axes[1].text(0.03, 0.88, "head amplified up to $\\times$20 — output core narrows",
             fontsize=6.6, color=V_DEEP, ha="left", va="top",
             transform=axes[1].transAxes)
axes[1].text(0.97, 0.05, "dashed line: human reference $\\sigma = 1$",
             fontsize=6.6, color=V_REFGRAY, ha="right", va="bottom",
             transform=axes[1].transAxes)
axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
save(fig, "F3")


# ---------------- F4: ECDF of pairwise profile Jaccard (Stage B) ----------------
BDIR = WORK / "repo" / "runs" / "b"
ENCDESC4 = [("a1", "a1 pitch-interval"), ("a2", "a2 rhythm"), ("a3", "a3 combined")]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.1))
for i, (ax, (a, desc)) in enumerate(zip(axes, ENCDESC4)):
    z = np.load(BDIR / f"jac.mtc.{a}.npz")
    j_in, j_out = z["j_in"], z["j_out"]
    for vals, lab, col, ls in ((j_in, "same-variant pairs", V_BLUE, "-"),
                               (j_out, "different-family pairs", V_RED, "--")):
        v = np.sort(vals)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=col, lw=1.5,
                drawstyle="steps-post", label=lab)
    rec = json.load(open(BDIR / f"mtc.{a}.json", encoding="utf-8"))
    b1, b2 = rec["b1"], rec["b2"]
    ax.grid(color="#dddddd", lw=0.5, alpha=0.7)
    ax.text(0.03, 0.97, f"({chr(65 + i)})", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=V_DEEP, ha="left", va="top")
    ax.text(0.03, 0.66, f"$\\Delta$ = {b1['contrast']:.3f}\n95% CI "
            f"[{b1['ci95'][0]:.3f}, {b1['ci95'][1]:.3f}]\nMantel r = {b2['mantel_r']:.2f}, p < .001",
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top", linespacing=1.35,
            color="#263238")
    ax.set_title(desc, fontsize=8.5)
    ax.set_xlim(-0.02, 0.72)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("ECDF", fontsize=8.5)
for ax in axes:
    ax.set_xlabel("pairwise profile Jaccard", fontsize=8)
fig.subplots_adjust(wspace=0.28)
save(fig, "figF4_ecdf")


# ---------------- F5: dose-response F(c) ----------------
doses = p3["doses"]
arms = [("M1_own", "M1 priming", "-", V_ORANGE),
        ("M2_own", "M2 in-context (own)", "-", V_BLUE),
        ("M2_cross", "M2 in-context (cross)", "--", V_GREEN),
        ("M3_filter", "M3 rejection control", ":", V_RED)]
fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
for j, (key, label) in enumerate(MODELS):
    for i, cult in enumerate(("CN", "EU")):
        ax = axes[i][j]
        for akey, alabel, ls, col in arms:
            s = p3["series"][f"{key}_{akey}_{cult}"]
            ax.plot(doses, s["F"], ls, color=col, lw=1.6, marker="o", ms=3.5,
                    label=alabel)
        ax.set_ylim(-0.05, 0.62)
        ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
        ax.text(0.04, 0.96, f"({chr(65 + i * 3 + j)})", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=V_DEEP, ha="left", va="top")
        own = p3["series"][f"{key}_M2_own_{cult}"]["F"]
        cross = p3["series"][f"{key}_M2_cross_{cult}"]["F"]
        gain = own[-1] - own[0]
        gap = own[-1] - cross[-1]
        ax.text(0.5, 0.97, f"own-culture gain {gain:+.2f}", transform=ax.transAxes,
                fontsize=6.6, color=V_BLUE, ha="center", va="top", fontweight="bold")
        if gap > 0.03:
            ax.text(0.5, 0.83, f"cross-culture seeding lags by {gap:.2f}",
                    transform=ax.transAxes, fontsize=6.0, color=V_GREEN,
                    ha="center", va="top")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.set_title(label, fontsize=9)
        if i == 1:
            ax.set_xlabel("coverage c", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("F(c)", fontsize=9)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.02),
           fontsize=7, frameon=False)
for i, cult in enumerate(("CN", "EU")):
    axes[i][0].annotate("Chinese-prompted" if cult == "CN" else "European-prompted",
                        xy=(-0.42, 0.5), xycoords="axes fraction",
                        fontsize=9, rotation=90, va="center")
save(fig, "F5")

print("ALL FIGURES DONE")
