# -*- coding: utf-8 -*-
"""Generate manuscript figures F1, F2, F3, F5 as SVG, then EMF via Inkscape.
Data sources: runs/d/p1_short23_results.json, runs/e/p2_results.json,
runs/f/p3_results.json. Output: Manuscript_work/figures/F{n}.svg/.emf
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


# ---------------- F1: pipeline schematic ----------------
fig, ax = plt.subplots(figsize=(6.6, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

def box(x, y, w, h, title, lines, fc="#eef2f7", ec="#3b5b7a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w/2, y + h - 3.2, title, ha="center", va="top",
            fontsize=8.5, fontweight="bold")
    ax.text(x + w/2, y + h - 7.0, "\n".join(lines), ha="center", va="top",
            fontsize=7.2, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color="#3b5b7a"))

box(1, 30, 20, 19, "Human corpora",
    ["MTC-FS-INST 2.0", "18,109 European tunes", "POP909", "Chinese pop piano"])
box(26, 30, 18, 19, "Symbolic encodings",
    ["a1  pitch-interval", "a2  rhythm", "a3  combined", "(transmission channel)"])
box(49, 30, 22, 19, "Frozen reference libraries",
    ["maximal itemset mining", "strata S1–S3 frozen", "ex ante, pre-generation", "(Supplement S1)"])
box(74, 30, 25, 19, "Machine channel",
    ["Models A / B / C", "Chinese- vs European-", "prompted, inference only", "zero parameter updates"])
arrow(21.4, 39.5, 25.6, 39.5); arrow(44.4, 39.5, 48.6, 39.5); arrow(71.4, 39.5, 73.6, 39.5)

box(1, 2, 30, 20, "P1  Cultural gap amplification",
    ["$\\Delta_D = D(Q,\\pi)_{out} - D(Q,\\pi)_{human}$", "permutation + size controls", "Fig. F2 \u00b7 Table T3"])
box(36, 2, 30, 20, "P2  Structured collapse",
    ["entropy $H(Q) < H(\\pi)$", "survival $\\sigma(S_1)>\\sigma(S_2)>\\sigma(S_3)$", "Fig. F3/F4 \u00b7 Table T4"])
box(71, 2, 28, 20, "P3  Seeding intervention",
    ["F(c) dose-response", "M1 priming \u00b7 M2 in-context", "M3 rejection control", "Fig. F5 \u00b7 Table T5"])
for x in (16, 51, 85):
    arrow(x, 29.6, x, 22.8)
save(fig, "F1")


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
enc_colors = {"a1": "#9db8cc", "a2": "#5d84a6", "a3": "#173f5f"}
for (enc, label, d, lo, hi, ref), y in zip(rows, ypos):
    c = enc_colors[enc]
    ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round")
    ax.plot(d, y, "o", color=c, ms=5)
# group separators + human-gap reference lines
bounds = {0: n, 1: n - 3, 2: n - 6, 3: 0}
for gi in range(3):
    ytop = n - 3 * gi - 0.5; ybot = n - 3 * (gi + 1) + 0.5
    ref = rows[gi * 3][5]
    ax.axvline(ref, ymin=(ybot + 0.5) / (n + 1), ymax=(ytop + 0.5) / (n + 1),
               color="#a33", ls=(0, (4, 3)), lw=1.0)
for i in (3, 6):
    ax.axhline(i - 0.5, color="#bbb", lw=0.6, ls=":")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{lab} \u00b7 {enc}" for enc, lab, *_ in rows], fontsize=8)
ax.axvline(0, color="#555", lw=0.8)
ax.set_xlabel("Amplified cultural gap  $\\Delta_D$  (95% bootstrap CI)", fontsize=9)
ax.set_xlim(-0.45, 1.05)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color="#a33", ls=(0, (4, 3)), lw=1.0,
                          label="human gap (scale-matched)")],
          loc="upper right", fontsize=8, frameon=False)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
save(fig, "F2")


# ---------------- F3: survival profiles sigma(S_k) ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
strata = ["S1 (head)", "S2 (middle)", "S3 (tail)"]
mk = {"Model A": ("o", "#173f5f"), "Model B": ("s", "#5d84a6"), "Model C": ("^", "#9db8cc")}
for ax, (cult, title) in zip(axes, [("CN", "Chinese-prompted output"),
                                    ("EU", "European-prompted output")]):
    for key, label in MODELS:
        vals = [p2["alphabets"]["a3"]["models"][key][cult]["T2_survival"][f"S{k}"]
                for k in (1, 2, 3)]
        m, c = mk[label]
        ax.plot(range(3), vals, marker=m, ms=5, lw=1.3, color=c, label=label)
    ax.axhline(1.0, color="#a33", ls=(0, (4, 3)), lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels(strata, fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.3, 2.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel("Survival rate $\\sigma(S_k)$  (log scale)", fontsize=9)
axes[1].text(0.03, 0.93, "reference rate $\\sigma=1$", fontsize=7, color="#a33",
             ha="left", va="top", transform=axes[1].transAxes)
axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
save(fig, "F3")


# ---------------- F5: dose-response F(c) ----------------
doses = p3["doses"]
arms = [("M1_own", "M1 priming", "-", "#777"),
        ("M2_own", "M2 in-context (own)", "-", "#173f5f"),
        ("M2_cross", "M2 in-context (cross)", "--", "#5d84a6"),
        ("M3_filter", "M3 rejection control", ":", "#a33")]
fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.0), sharex=True)
for j, (key, label) in enumerate(MODELS):
    for i, cult in enumerate(("CN", "EU")):
        ax = axes[i][j]
        for akey, alabel, ls, col in arms:
            s = p3["series"][f"{key}_{akey}_{cult}"]
            ax.plot(doses, s["F"], ls, color=col, lw=1.3, marker="o", ms=3.5,
                    label=alabel)
        ax.set_ylim(-0.05, 0.62)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i == 0:
            ax.set_title(label, fontsize=9)
        if i == 1:
            ax.set_xlabel("coverage c", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("F(c)", fontsize=9)
axes[0][2].legend(fontsize=6.8, frameon=False, loc="upper left")
for i, cult in enumerate(("CN", "EU")):
    axes[i][0].annotate("Chinese-prompted" if cult == "CN" else "European-prompted",
                        xy=(-0.42, 0.5), xycoords="axes fraction",
                        fontsize=9, rotation=90, va="center")
save(fig, "F5")

print("ALL FIGURES DONE")
