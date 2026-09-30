#!/usr/bin/env python3
"""
Acylcarnitine/CoA uptake analysis, quantitative + qualitative combined.

Per genotype, a darker bar = Quant., a lighter bar = Qual., grouped
side by side (Col-0 quant | Col-0 qual || bou quant | bou qual).
Genotype identity is carried by edge color (black = Col-0, dark red =
bou); treatment is carried by hatch pattern.

Statistics are computed separately for Quant. and Qual. (independent
measurements) and saved as Uptake_Stats_Quan.csv / Uptake_Stats_Qual.csv.

Output figures:
  Figure_AB_Carnitine_merged — 13C-Octanoyl-carnitine (top) +
                                13C-Palmitoyl-carnitine (bottom),
                                all 5 treatments, broken y-axis.
  Figure_AB_CoA_merged       — 13C-Octanoyl-CoA (top) +
                                13C-Palmitoyl-CoA (bottom),
                                all 5 treatments, unbroken y-axis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Patch as MPatch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

EXCEL_FILE = "Uptake_assay.xlsx"

TREATMENT_SHORT = {
    1: "+CoA +Car",
    2: "+CoA −Car",
    3: "−CoA +Car",
    4: "−CoA −Car",
    5: "ETO +CoA +Car",
}

COL0_FILL  = "#707071"
COL0_EDGE  = "black"
BOU_FILL   = "#C03830"
BOU_EDGE   = "#4f0501"
COL0_JITT  = "#adadad"
BOU_JITT   = "#f5a8a4"


def _lighten(hex_color, factor):
    """Blend a hex color toward white by `factor` (0-1), for the Qual. shade."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


COL0_QUANT_FILL = COL0_FILL
COL0_QUAL_FILL  = _lighten(COL0_FILL, 0.55)
BOU_QUANT_FILL  = BOU_FILL
BOU_QUAL_FILL   = _lighten(BOU_FILL, 0.45)

FILLS = {
    ("Col-0", "Quan"): COL0_QUANT_FILL,
    ("Col-0", "Qual"): COL0_QUAL_FILL,
    ("bou",   "Quan"): BOU_QUANT_FILL,
    ("bou",   "Qual"): BOU_QUAL_FILL,
}
EDGES    = {"Col-0": COL0_EDGE, "bou": BOU_EDGE}
DOT_FACE = {"Col-0": COL0_JITT, "bou": BOU_JITT}

GENOTYPES  = ["Col-0", "bou"]
TREATMENTS = [1, 2, 3, 4, 5]
TREATMENT_HATCHES = {1: "", 2: "///", 3: "...", 4: "xxx", 5: "\\\\"}
SUBTYPES   = ["Quan", "Qual"]   # left-to-right order within a genotype

JITTER_SIZE = 34
FONT_TITLE  = 16
FONT_LABEL  = 12
FONT_TICK   = 12
FONT_LEGEND = 11
DPI         = 1000

BAR_W        = 0.20   # single-type panel bar width
BAR_W_MERGED = 0.15   # narrower, so 4 bars/treatment fit in the merged panels


def x_positions(treatments, genotypes, tight=False):
    """2-bars-per-treatment layout (Col-0, bou)."""
    geno_gap = BAR_W * 1.5
    group_gap = BAR_W * (2.8 if tight else 4.5)
    pos = {}
    for t_idx, treat in enumerate(treatments):
        cx = t_idx * group_gap
        for g_idx, geno in enumerate(genotypes):
            pos[(treat, geno)] = cx + (g_idx - (len(genotypes) - 1) / 2) * geno_gap
    return pos


def x_positions_merged(treatments, genotypes, subtypes, tight=False):
    """4-bars-per-treatment layout: two genotype sub-groups, each with a
    Quant. bar then a Qual. bar side by side."""
    sub_gap   = BAR_W_MERGED * 1.15
    geno_gap  = BAR_W_MERGED * 3.0
    group_gap = BAR_W_MERGED * (6.0 if tight else 9.5)
    pos = {}
    for t_idx, treat in enumerate(treatments):
        cx = t_idx * group_gap
        for g_idx, geno in enumerate(genotypes):
            geno_center = cx + (g_idx - (len(genotypes) - 1) / 2) * geno_gap
            for s_idx, sub in enumerate(subtypes):
                pos[(treat, geno, sub)] = (
                    geno_center + (s_idx - (len(subtypes) - 1) / 2) * sub_gap)
    return pos


print("Loading data...")
df_car = pd.read_excel(EXCEL_FILE, sheet_name="Carnitine")
df_coa = pd.read_excel(EXCEL_FILE, sheet_name="CoA")
for df in [df_car, df_coa]:
    df.columns = df.columns.str.strip()
    df["Genotype"] = df["Genotype"].str.strip()
    df["Compound"] = df["Compound"].str.strip()
df_all = pd.concat([df_car, df_coa], ignore_index=True)


def auc_col(compound, sub):
    """Resolve the AUC column name for a compound/measurement type
    (kept as a function rather than inlined, in case a compound-specific
    override is ever needed)."""
    return f"AUC_{sub}"


def get_vals(df, compound, treatment, genotype, col):
    mask = ((df["Compound"] == compound) &
            (df["Treatment"] == treatment) &
            (df["Genotype"] == genotype))
    return df.loc[mask, col].dropna().values


def welch_t(a, b):
    a = np.array(a, dtype=float)[~np.isnan(np.array(a, dtype=float))]
    b = np.array(b, dtype=float)[~np.isnan(np.array(b, dtype=float))]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return stats.ttest_ind(a, b, equal_var=False)


def sig_label(p):
    if np.isnan(p): return "n.d."
    if p >= 0.05:   return "ns"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"


PLANNED = [
    ("O-carnitine",1,"Col-0",2,"Col-0"), ("O-carnitine",1,"Col-0",3,"Col-0"),
    ("O-carnitine",1,"Col-0",4,"Col-0"), ("O-carnitine",1,"Col-0",5,"Col-0"),
    ("O-carnitine",1,"bou",  2,"bou"),   ("O-carnitine",1,"bou",  3,"bou"),
    ("O-carnitine",1,"bou",  4,"bou"),   ("O-carnitine",1,"bou",  5,"bou"),
    ("O-carnitine",1,"Col-0",1,"bou"),   ("O-carnitine",1,"Col-0",2,"bou"),
    ("O-carnitine",1,"Col-0",3,"bou"),   ("O-carnitine",1,"Col-0",4,"bou"),
    ("O-carnitine",1,"Col-0",5,"bou"),

    ("P-carnitine",1,"Col-0",2,"Col-0"), ("P-carnitine",1,"Col-0",3,"Col-0"),
    ("P-carnitine",1,"Col-0",4,"Col-0"), ("P-carnitine",1,"Col-0",5,"Col-0"),
    ("P-carnitine",1,"bou",  2,"bou"),   ("P-carnitine",1,"bou",  3,"bou"),
    ("P-carnitine",1,"bou",  4,"bou"),   ("P-carnitine",1,"bou",  5,"bou"),
    ("P-carnitine",1,"Col-0",1,"bou"),   ("P-carnitine",1,"Col-0",2,"bou"),
    ("P-carnitine",1,"Col-0",3,"bou"),   ("P-carnitine",1,"Col-0",4,"bou"),
    ("P-carnitine",1,"Col-0",5,"bou"),

    ("O-CoA",1,"Col-0",2,"Col-0"),       ("O-CoA",1,"Col-0",3,"Col-0"),
    ("O-CoA",1,"Col-0",4,"Col-0"),       ("O-CoA",1,"Col-0",5,"Col-0"),
    ("O-CoA",1,"bou",  2,"bou"),         ("O-CoA",1,"bou",  3,"bou"),
    ("O-CoA",1,"bou",  4,"bou"),         ("O-CoA",1,"bou",  5,"bou"),
    ("O-CoA",1,"Col-0",1,"bou"),         ("O-CoA",1,"Col-0",2,"bou"),
    ("O-CoA",1,"Col-0",3,"bou"),         ("O-CoA",1,"Col-0",4,"bou"),
    ("O-CoA",1,"Col-0",5,"bou"),

    ("P-CoA",1,"Col-0",2,"Col-0"),       ("P-CoA",1,"Col-0",3,"Col-0"),
    ("P-CoA",1,"Col-0",4,"Col-0"),       ("P-CoA",1,"Col-0",5,"Col-0"),
    ("P-CoA",1,"bou",  2,"bou"),         ("P-CoA",1,"bou",  3,"bou"),
    ("P-CoA",1,"bou",  4,"bou"),         ("P-CoA",1,"bou",  5,"bou"),
    ("P-CoA",1,"Col-0",1,"bou"),         ("P-CoA",1,"Col-0",2,"bou"),
    ("P-CoA",1,"Col-0",3,"bou"),         ("P-CoA",1,"Col-0",4,"bou"),
    ("P-CoA",1,"Col-0",5,"bou"),
]
n_comp = len(PLANNED)


def run_statistics(sub, label):
    """Runs the LOQ + Welch t-test + Bonferroni pipeline for one AUC
    type ("Quan" or "Qual") and saves Uptake_Stats_{label}.csv."""

    def compute_loq(compound, blank_treatment=4, genotype="bou"):
        vals = get_vals(df_all, compound, blank_treatment, genotype,
                        auc_col(compound, sub))
        if len(vals) < 2:
            return 0.0
        return float(np.mean(vals) + 100 * np.std(vals, ddof=1))

    LOQ = {c: compute_loq(c) for c in
           ["O-carnitine", "P-carnitine", "O-CoA", "P-CoA"]}
    print(f"\nLOQ thresholds ({label}, AUC units):")
    for k, v in LOQ.items():
        print(f"  {k}: {v:.1f}")

    stat_records = []
    for (cmpd, t1, g1, t2, g2) in PLANNED:
        col = auc_col(cmpd, sub)
        a   = get_vals(df_all, cmpd, t1, g1, col)
        b   = get_vals(df_all, cmpd, t2, g2, col)
        loq = LOQ.get(cmpd, 0.0)

        mean1 = float(np.mean(a)) if len(a) else np.nan
        sd1   = float(np.std(a, ddof=1)) if len(a) > 1 else np.nan
        mean2 = float(np.mean(b)) if len(b) else np.nan
        sd2   = float(np.std(b, ddof=1)) if len(b) > 1 else np.nan

        if mean1 < loq and mean2 < loq:
            stat_records.append({
                "Compound": cmpd, "Group1": f"T{t1} {g1}", "n1": len(a),
                "Mean1": round(mean1, 4) if not np.isnan(mean1) else np.nan,
                "SD1":   round(sd1, 4)   if not np.isnan(sd1)   else np.nan,
                "Group2": f"T{t2} {g2}", "n2": len(b),
                "Mean2": round(mean2, 4) if not np.isnan(mean2) else np.nan,
                "SD2":   round(sd2, 4)   if not np.isnan(sd2)   else np.nan,
                "LOQ": round(loq, 4),
                "Group1_above_LOQ": "NO", "Group2_above_LOQ": "NO",
                "p_raw": np.nan, "p_Bonferroni": np.nan,
                "Sig": "ND — both below LOQ",
            })
            continue

        _, p = welch_t(a, b)
        p_bonf = min(p * 13, 1.0) if not np.isnan(p) else np.nan
        stat_records.append({
            "Compound": cmpd, "Group1": f"T{t1} {g1}", "n1": len(a),
            "Mean1": round(mean1, 4) if not np.isnan(mean1) else np.nan,
            "SD1":   round(sd1, 4)   if not np.isnan(sd1)   else np.nan,
            "Group2": f"T{t2} {g2}", "n2": len(b),
            "Mean2": round(mean2, 4) if not np.isnan(mean2) else np.nan,
            "SD2":   round(sd2, 4)   if not np.isnan(sd2)   else np.nan,
            "LOQ": round(loq, 4),
            "Group1_above_LOQ": "YES" if mean1 >= loq else "NO",
            "Group2_above_LOQ": "YES" if mean2 >= loq else "NO",
            "p_raw": round(p, 8) if not np.isnan(p) else np.nan,
            "p_Bonferroni": round(p_bonf, 8) if not np.isnan(p_bonf) else np.nan,
            "Sig": sig_label(p_bonf),
        })

    stats_df = pd.DataFrame(stat_records)
    stats_df.to_csv(f"Uptake_Stats_{label}.csv", index=False)
    print(f"\nStatistics saved: Uptake_Stats_{label}.csv ({len(stats_df)} comparisons)")
    nd_count = (stats_df["Sig"].str.startswith("ND")).sum()
    print(f"  Skipped (both below LOQ): {nd_count}")
    print(f"  Tested (at least one above LOQ): {len(stats_df) - nd_count}")
    print(f"\nKey results ({label}):")
    tested = stats_df[~stats_df["Sig"].str.startswith("ND")]
    print(tested[["Compound", "Group1", "Group2", "Mean1", "Mean2",
                  "p_Bonferroni", "Sig"]].to_string(index=False))
    return LOQ


print("=" * 65)
print("ACYLCARNITINE UPTAKE — QUANTITATIVE + QUALITATIVE STATISTICS")
print("=" * 65)
LOQ_QUAN = run_statistics("Quan", "Quan")
LOQ_QUAL = run_statistics("Qual", "Qual")


def draw_bars(ax, compound, df, treatments, genotypes, xpos, col):
    """Single-type (Quant. or Qual. only) bar panel."""
    colors = {
        "Col-0": {"fill": COL0_FILL, "edge": COL0_EDGE, "dot_face": COL0_JITT},
        "bou":   {"fill": BOU_FILL,  "edge": BOU_EDGE,  "dot_face": BOU_JITT},
    }
    for treat in treatments:
        for geno in genotypes:
            vals   = get_vals(df, compound, treat, geno, col)
            mean_v = np.mean(vals) if len(vals) else 0
            std_v  = np.std(vals, ddof=1) if len(vals) > 1 else 0
            xp     = xpos[(treat, geno)]
            c      = colors[geno]

            ax.bar(xp, mean_v, BAR_W * 0.92,
                   facecolor=c["fill"], edgecolor=c["edge"],
                   linewidth=3.5, hatch=TREATMENT_HATCHES[treat], alpha=1.0)
            ax.errorbar(xp, mean_v, yerr=std_v, fmt="none", color=c["edge"],
                        capsize=3.5, linewidth=1.3, capthick=1.3)
            jitter = np.random.uniform(-BAR_W * 0.18, BAR_W * 0.18, len(vals))
            ax.scatter(xp + jitter, vals, facecolors=c["dot_face"],
                       edgecolors="black", s=JITTER_SIZE, zorder=5, linewidth=1.3)


def simple_bar_panel(ax, compound, df, treatments, genotypes, col, ylabel="AUC (a.u.)"):
    xpos = x_positions(treatments, genotypes)
    x_centers = [np.mean([xpos[(t, g)] for g in genotypes]) for t in treatments]

    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    draw_bars(ax, compound, df, treatments, genotypes, xpos, col)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([TREATMENT_SHORT[t] for t in treatments],
                       fontsize=FONT_TICK, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FONT_TICK)
    ax.tick_params(labelsize=FONT_TICK)

    margin = BAR_W * 1.0
    x_all = [xpos[(t, g)] for t in treatments for g in genotypes]
    ax.set_xlim(min(x_all) - margin, max(x_all) + margin)
    return xpos


def draw_bars_merged(ax, compound, df, treatments, genotypes, subtypes, xpos):
    """Quant. + Qual. bars side by side, for the merged figures."""
    for treat in treatments:
        for geno in genotypes:
            for sub in subtypes:
                col    = auc_col(compound, sub)
                vals   = get_vals(df, compound, treat, geno, col)
                mean_v = np.mean(vals) if len(vals) else 0
                std_v  = np.std(vals, ddof=1) if len(vals) > 1 else 0
                xp     = xpos[(treat, geno, sub)]
                fill   = FILLS[(geno, sub)]
                edge   = EDGES[geno]

                ax.bar(xp, mean_v, BAR_W_MERGED * 0.92,
                       facecolor=fill, edgecolor=edge,
                       linewidth=2.6, hatch=TREATMENT_HATCHES[treat], alpha=1.0)
                ax.errorbar(xp, mean_v, yerr=std_v, fmt="none", color=edge,
                            capsize=2.8, linewidth=1.1, capthick=1.1)
                jitter = np.random.uniform(-BAR_W_MERGED * 0.18,
                                            BAR_W_MERGED * 0.18, len(vals))
                ax.scatter(xp + jitter, vals, facecolors=DOT_FACE[geno],
                           edgecolors="black", s=JITTER_SIZE, zorder=5, linewidth=1.0)


def simple_bar_panel_merged(ax, compound, df, treatments, genotypes, subtypes,
                            ylabel="AUC (a.u.)"):
    """Single-axis (no break) Quant.+Qual. panel, for compounds whose
    dynamic range doesn't need a broken y-axis."""
    xpos = x_positions_merged(treatments, genotypes, subtypes)
    x_centers = [np.mean([xpos[(t, g, s)] for g in genotypes for s in subtypes])
                 for t in treatments]

    ax.set_facecolor("white"); ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    draw_bars_merged(ax, compound, df, treatments, genotypes, subtypes, xpos)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([TREATMENT_SHORT[t] for t in treatments],
                       fontsize=FONT_TICK, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FONT_TICK)
    ax.tick_params(labelsize=FONT_TICK)

    margin = BAR_W_MERGED * 1.5
    x_all = [xpos[(t, g, s)] for t in treatments for g in genotypes for s in subtypes]
    ax.set_xlim(min(x_all) - margin, max(x_all) + margin)

    all_vals = []
    for t in treatments:
        for g in genotypes:
            for sub in subtypes:
                all_vals.extend(get_vals(df, compound, t, g, auc_col(compound, sub)))
    y_max = max(all_vals) * 1.35 if len(all_vals) else 1.0
    ax.set_ylim(0, y_max)
    return xpos


def plot_broken_bars_merged(ax_top, ax_bot, compound, df, treatments, genotypes,
                            subtypes, top_ylim, bot_ylim):
    xpos = x_positions_merged(treatments, genotypes, subtypes, tight=True)
    x_centers = [np.mean([xpos[(t, g, s)] for g in genotypes for s in subtypes])
                 for t in treatments]

    for ax, ylim in [(ax_top, top_ylim), (ax_bot, bot_ylim)]:
        ax.set_facecolor("white"); ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        draw_bars_merged(ax, compound, df, treatments, genotypes, subtypes, xpos)
        ax.set_ylim(ylim)
        ax.set_xticks(x_centers)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK)

    ax_bot.set_xticklabels([TREATMENT_SHORT[t] for t in treatments],
                           fontsize=FONT_TICK, fontweight="bold")
    ax_top.set_xticklabels([])

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    ax_bot.yaxis.set_offset_position("left")

    ax_top.set_ylabel("AUC (a.u.)", fontsize=FONT_LABEL, fontweight="bold", labelpad=14)
    ax_bot.set_ylabel("")
    ax_bot.yaxis.get_offset_text().set_position((-0.08, -2.30))

    margin = BAR_W_MERGED * 1.5
    x_all = [xpos[(t, g, s)] for t in treatments for g in genotypes for s in subtypes]
    xlim = (min(x_all) - margin, max(x_all) + margin)
    ax_top.set_xlim(xlim); ax_bot.set_xlim(xlim)
    return xpos, x_centers


def auto_ylims_merged(compound, pos_treats=[1, 5], neg_treats=[2, 3, 4]):
    """Y-limits pooling both AUC_Quan and AUC_Qual so neither is clipped.
    The break point is set from whichever is taller: the background-bar
    magnitude (with headroom) or the Qual. bar at the positive
    treatment(s), so that bar sits fully under the break."""
    quant_pos_vals, qual_pos_vals, bg_vals = [], [], []
    for t in pos_treats:
        for g in GENOTYPES:
            quant_pos_vals.extend(get_vals(df_all, compound, t, g, auc_col(compound, "Quan")))
            qual_pos_vals.extend(get_vals(df_all, compound, t, g, auc_col(compound, "Qual")))
    for t in neg_treats:
        for g in GENOTYPES:
            for sub in SUBTYPES:
                bg_vals.extend(get_vals(df_all, compound, t, g, auc_col(compound, sub)))

    top_max = max(quant_pos_vals + qual_pos_vals) * 1.45 if (quant_pos_vals or qual_pos_vals) else 1e6

    bg_bot_max   = max(bg_vals) * 2.2 if bg_vals else 1000
    qual_bot_max = max(qual_pos_vals) * 1.25 if qual_pos_vals else 0
    bot_max = max(bg_bot_max, qual_bot_max)

    # Clamp so the top panel's floor never reaches or exceeds its
    # ceiling, which would otherwise invert the axis.
    if bot_max * 1.20 >= top_max:
        bot_max = top_max / 1.30

    return (bot_max * 1.20, top_max), (0, bot_max)


geno_subtype_legend = [
    MPatch(facecolor=COL0_QUANT_FILL, edgecolor=COL0_EDGE, linewidth=1.4, label="Col-0 — Quant."),
    MPatch(facecolor=COL0_QUAL_FILL,  edgecolor=COL0_EDGE, linewidth=1.4, label="Col-0 — Qual."),
    MPatch(facecolor=BOU_QUANT_FILL,  edgecolor=BOU_EDGE,  linewidth=1.4, label="bou — Quant."),
    MPatch(facecolor=BOU_QUAL_FILL,   edgecolor=BOU_EDGE,  linewidth=1.4, label="bou — Qual."),
]
geno_legend = [
    MPatch(facecolor=COL0_FILL, edgecolor=COL0_EDGE, linewidth=1.4, label="Col-0"),
    MPatch(facecolor=BOU_FILL,  edgecolor=BOU_EDGE,  linewidth=1.4, label="bou"),
]
hatch_legend = [
    MPatch(facecolor="#A0A0A0", hatch=TREATMENT_HATCHES[t], edgecolor="white",
           label=TREATMENT_SHORT[t])
    for t in TREATMENTS
]

# Figure 1 -- O-carnitine + P-carnitine, merged, broken y-axis
fig1 = plt.figure(figsize=(16, 14.5))
fig1.patch.set_facecolor("white")
outer1 = gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[1, 1], hspace=0.55, top=0.90)

gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer1[0],
                                        height_ratios=[2.5, 1], hspace=0.08)
ax_a_top = fig1.add_subplot(gs_a[0])
ax_a_bot = fig1.add_subplot(gs_a[1])

top_ylim_o, bot_ylim_o = auto_ylims_merged("O-carnitine")
plot_broken_bars_merged(ax_a_top, ax_a_bot, "O-carnitine", df_all,
                        TREATMENTS, GENOTYPES, SUBTYPES, top_ylim_o, bot_ylim_o)
ax_a_top.set_title("¹³C-Octanoyl-carnitine", fontsize=FONT_TITLE, fontweight="bold", pad=10)

gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer1[1],
                                        height_ratios=[2.5, 1], hspace=0.08)
ax_b_top = fig1.add_subplot(gs_b[0])
ax_b_bot = fig1.add_subplot(gs_b[1])

top_ylim_p, bot_ylim_p = auto_ylims_merged("P-carnitine")
plot_broken_bars_merged(ax_b_top, ax_b_bot, "P-carnitine", df_all,
                        TREATMENTS, GENOTYPES, SUBTYPES, top_ylim_p, bot_ylim_p)
ax_b_top.set_title("¹³C-Palmitoyl-carnitine", fontsize=FONT_TITLE, fontweight="bold", pad=10)

# One shared legend for the whole figure: treatment (hatch) row, then
# genotype x Quant./Qual. (color) row, both centered above the panels.
leg_treat = fig1.legend(handles=hatch_legend, fontsize=FONT_LEGEND, loc="upper center",
                        bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, borderpad=0.6)
fig1.add_artist(leg_treat)
fig1.legend(handles=geno_subtype_legend, fontsize=FONT_LEGEND, loc="upper center",
           bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)

fig1.text(0.5, 0.01, "Treatment", ha="center", fontsize=FONT_LABEL + 1, fontweight="bold")

plt.savefig("Figure_AB_Carnitine_merged.png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("Figure_AB_Carnitine_merged.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved: Figure_AB_Carnitine_merged.png / .pdf")
plt.show()

# Figure 2 -- O-CoA + P-CoA, merged, unbroken y-axis
fig2 = plt.figure(figsize=(16, 12))
fig2.patch.set_facecolor("white")
outer2 = gridspec.GridSpec(2, 1, figure=fig2, height_ratios=[1, 1], hspace=0.45, top=0.90)

ax_a2 = fig2.add_subplot(outer2[0])
simple_bar_panel_merged(ax_a2, "O-CoA", df_all, TREATMENTS, GENOTYPES, SUBTYPES,
                        ylabel="AUC (a.u.)")
ax_a2.set_title("¹³C-Octanoyl-CoA", fontsize=FONT_TITLE, fontweight="bold", pad=10)

ax_b2 = fig2.add_subplot(outer2[1])
simple_bar_panel_merged(ax_b2, "P-CoA", df_all, TREATMENTS, GENOTYPES, SUBTYPES,
                        ylabel="AUC (a.u.)")
ax_b2.set_title("¹³C-Palmitoyl-CoA", fontsize=FONT_TITLE, fontweight="bold", pad=10)

leg_treat2 = fig2.legend(handles=hatch_legend, fontsize=FONT_LEGEND, loc="upper center",
                         bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False, borderpad=0.6)
fig2.add_artist(leg_treat2)
fig2.legend(handles=geno_subtype_legend, fontsize=FONT_LEGEND, loc="upper center",
           bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)

fig2.text(0.5, 0.01, "Treatment", ha="center", fontsize=FONT_LABEL + 1, fontweight="bold")

plt.savefig("Figure_AB_CoA_merged.png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("Figure_AB_CoA_merged.pdf", bbox_inches="tight", facecolor="white")
print("Saved: Figure_AB_CoA_merged.png / .pdf")
plt.show()
