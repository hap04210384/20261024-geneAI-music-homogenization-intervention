# -*- coding: utf-8 -*-
"""Generate manuscript figures F1, F2, F3, F5 as SVG, then EMF via Inkscape.
Data sources: runs/d/p1_short23_results.json, runs/e/p2_results.json,
runs/f/p3_results.json. Output: Manuscript_work/figures/F{n}.svg/.emf
"""
import json
import subprocess
from pathlib import Path

import numpy as np
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

# unified palette: deep navy -> vivid mid blue -> light sky, warm red accent
C_DARK, C_MID, C_LIGHT = "#154360", "#2e86c1", "#85c1e9"
C_RED, C_GRAY = "#c0392b", "#707b7c"
C_SHADE, C_AMBER = "#fdebd0", "#9c4a08"

# vivid unified palette for F2-F5 (consistent semantics across figures):
# deep blue = primary emphasis / a3 / own-culture arm; vivid blue = main series;
# azure/sky = lighter encodings; cyan = cross-culture dashed; purple/green =
# secondary categories; red = human reference / control / contrast
V_DEEP, V_BLUE, V_AZURE = "#0D47A1", "#1565C0", "#1E88E5"
V_SKY, V_CYAN = "#64B5F6", "#4FC3F7"
V_PURPLE, V_GREEN, V_RED = "#8E24AA", "#43A047", "#E53935"
V_AMBER = "#B3540A"

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
fig, ax = plt.subplots(figsize=(6.6, 4.36))
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

def rbox(x, y, w, h, fc="#f7fafc", ec="#9db6c9", lw=0.9, rs=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={rs}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def titled_box(x, y, w, h, title, lines, tfs=7.4, fs=6.2, tcol=C_DARK):
    rbox(x, y, w, h)
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=tcol, zorder=4)
    ax.text(x + w / 2, y + h - 4.3, "\n".join(lines), ha="center", va="top",
            fontsize=fs, color="#1c2b3a", linespacing=1.45, zorder=4)

def result_card(x, y, w, h, tag, title, lines, ref, accent):
    rbox(x, y, w, h)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x + 0.35, y + h - 4.5), w - 0.7, 4.0, fc=accent, ec=accent, zorder=3))
    ax.text(x + 1.3, y + h - 2.5, f"{tag} \u00b7 {title}", fontsize=6.0,
            fontweight="bold", color="white", va="center", zorder=4)
    ax.text(x + w / 2, y + h - 6.3, "\n".join(lines), ha="center", va="top",
            fontsize=6.2, color="#1c2b3a", linespacing=1.5, zorder=4)
    ax.text(x + w / 2, y + 1.3, ref, ha="center", va="bottom",
            fontsize=6.0, color="#5d6f80", zorder=4)

def arrow(x1, y1, x2, y2, color=C_DARK, lw=1.1, ls="-", ms=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, zorder=6))

# ---------------- row 1: preparation pipeline ----------------
titled_box(2, 53.5, 26, 10.5, "HUMAN CORPORA",
           ["MTC-FS-INST 2.0 \u00b7 18,109 European",
            "folk tunes (NL \u00b7 BE)",
            "POP909 \u00b7 909 Chinese pop songs"], fs=6.0)

ax.text(38, 64.6, "ENCODINGS \u2014 the transmission channel", ha="center",
        fontsize=6.4, fontweight="bold", color=C_MID)
for yy, lab in ((59.2, "a1  pitch\u2013interval"), (55.2, "a2  rhythm"), (51.2, "a3  joint")):
    rbox(31.5, yy, 13, 3.4, fc="#eaf2f8", ec=C_MID, lw=0.9)
    ax.text(38, yy + 1.7, lab, ha="center", va="center", fontsize=6.2, color=C_DARK, zorder=4)

titled_box(48, 53.5, 27, 10.5, "FROZEN REFERENCE LIBRARIES",
           ["maximal itemset mining", "strata S1 head \u00b7 S2 \u00b7 S3 tail",
            "frozen ex ante \u2014 pre-generation", "public release + checksums (Suppl. S1)"], fs=6.0)

titled_box(79, 53.5, 19, 10.5, "CONTROLS & RULES",
           ["permutation + bootstrap CIs", "size-matched human gaps",
            "null models N1\u2013N3", "manipulation gate (2/3)"], fs=5.9, tcol=C_RED)

arrow(28.2, 58.7, 31.2, 58.7)
arrow(44.7, 58.7, 47.7, 58.7)
arrow(75.2, 58.7, 78.7, 58.7, color="#8a9bab", lw=0.9, ls=(0, (2, 2)))

# ---------------- row 2: generation through the bottleneck ----------------
titled_box(2, 30, 21, 17, "MACHINE CHANNEL",
           ["Models A \u00b7 B \u00b7 C", "(three families)", "prompted per culture",
            "inference only \u2014", "zero parameter updates"], fs=6.2)
arrow(23.2, 38.5, 29.4, 38.5)
ax.text(26.3, 35.4, "CN / EU prompts", ha="center", fontsize=6.0,
        style="italic", color="#5d6f80")

from matplotlib.patches import Polygon
ax.add_patch(Polygon([(30, 47), (50, 40), (50, 35), (30, 28)],
                     closed=True, fc="#d7e8f5", ec=C_MID, lw=1.3, zorder=2))
ax.text(40, 25.0, "one shared training bottleneck", ha="center", fontsize=6.8,
        fontweight="bold", color=C_DARK)
ax.text(40, 49.9, "variance shrinkage where the idiom is learned \u00b7 "
                  "off-manifold drift where it is not",
        ha="center", fontsize=6.0, style="italic", color="#5d6f80")

arrow(50.3, 37.5, 53.7, 37.5)
titled_box(54, 30, 44, 16, "CONVERGENT OUTPUT SPACE Q",
           ["surface style survives \u2014 recognition 0.68\u20130.76",
            "combinatorial vocabulary lost \u2014 entropy \u221243%",
            "88% of frozen strata extinct from generated output",
            "dist(Q, \u03c0_EU) \u2248 dist(Q, \u03c0_CN): one generic space"], fs=6.3)

arrow(60, 53.2, 60, 46.5, color="#8a9bab", lw=0.9)
arrow(76, 29.6, 76, 22.0)

# ---------------- row 3: the three propositions ----------------
result_card(2, 2, 31, 19.5, "P1", "CULTURAL GAP AMPLIFICATION",
            ["$\Delta_D = D(Q,\pi)_{out} - D(Q,\pi)_{human}$",
             "up to +0.70 at encoding a3",
             "(human cultural gap: 0.978)",
             "outputs depart further from the",
             "frozen library than human music"],
            "Fig. 2 \u00b7 Table 2", C_DARK)
result_card(34.5, 2, 31, 19.5, "P2", "STRUCTURED COLLAPSE",
            ["$\u03c3(S_1) \u226b \u03c3(S_2) \u2248 \u03c3(S_3) \u2248 0$",
             "the tail dies first \u2014 wholesale",
             "extinction of the non-core",
             "vocabulary, not graded loss"],
            "Fig. 3/4 \u00b7 Table 3", C_MID)
result_card(67, 2, 31, 19.5, "P3", "ZERO-TRAINING REPAIR",
            ["input seeding: +0.33, dose-dependent",
             "output filtering: +0.00",
             "the channel is not refusing these",
             "cultures \u2014 it is not being shown them"],
            "Fig. 5 \u00b7 Table 5", C_RED)
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
enc_colors = {"a1": V_SKY, "a2": V_AZURE, "a3": V_DEEP}
for (enc, label, d, lo, hi, ref), y in zip(rows, ypos):
    c = enc_colors[enc]
    ax.plot([lo, hi], [y, y], color=c, lw=1.6, solid_capstyle="round")
    ax.plot(d, y, "o", color=c, ms=5.5)
# compression zone shading (Delta_D > 0) + zero line
ax.axvspan(0, 1.38, color=C_SHADE, zorder=0)
ax.text(0.40, n - 0.7, "compression zone ($\\Delta_D > 0$): outputs converge\ncloser to each other than same-culture human music does",
        fontsize=6.6, color=V_AMBER, va="top", ha="left", linespacing=1.3)
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
mk = {"Model A": ("o", V_BLUE), "Model B": ("s", V_PURPLE), "Model C": ("^", V_GREEN)}
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
    ax.axhline(1.0, color=V_RED, ls=(0, (4, 3)), lw=1.2)
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
             fontsize=6.6, color=V_RED, ha="right", va="bottom",
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
arms = [("M1_own", "M1 priming", "-", V_PURPLE),
        ("M2_own", "M2 in-context (own)", "-", V_DEEP),
        ("M2_cross", "M2 in-context (cross)", "--", V_CYAN),
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
                fontsize=6.6, color=V_DEEP, ha="center", va="top", fontweight="bold")
        if gap > 0.03:
            ax.text(0.5, 0.83, f"cross-culture seeding lags by {gap:.2f}",
                    transform=ax.transAxes, fontsize=6.0, color="#0288D1",
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
