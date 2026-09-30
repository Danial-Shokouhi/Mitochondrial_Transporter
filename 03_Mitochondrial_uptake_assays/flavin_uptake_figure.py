#!/usr/bin/env python3
"""
Flavin uptake assay — FAD transport into plant mitochondria.
Col-0 vs bou mutant | 10 min and 30 min incubation | 4 replicates.

Produces:
  Figure 1: Kinetic uptake line plot — FAD
  Flavin_Stats.csv — all statistical comparisons
  Flavin_Transport_Rate_per_replicate.csv / _summary.csv — transport
  rate, (Net_uptake@30min - Net_uptake@10min) / 20 min

Input: Flavin_uptake.xlsx, sheet "Net_uptake"
  Columns: Genotype | Flavin | Incubation_time | Replicate | Net_uptake
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

EXCEL_FILE  = "Flavin_uptake.xlsx"
SHEET_NAME  = "Net_uptake"
DPI         = 1000
FONT_TITLE  = 13
FONT_LABEL  = 12
FONT_TICK   = 10.5
FONT_ANNOT  = 9.5
FONT_LEGEND = 15

COL0_COLOR  = "black"
COL0_FILL   = "#707071"
BOU_COLOR   = "#4f0501"
BOU_FILL    = "#C03830"

GENOTYPES   = ["Col-0", "bou"]
FLAVINS     = ["FAD"]
TIMEPOINTS  = [10, 30]
JITTER_SIZE = 38

print("Loading data...")
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()
df["Genotype"]        = df["Genotype"].str.strip()
df["Flavin"]          = df["Flavin"].str.strip()
df["Incubation_time"] = df["Incubation_time"].astype(int)
print(f"  Rows: {len(df)}")
print(df.groupby(["Genotype", "Flavin", "Incubation_time"])["Net_uptake"].describe())


def get_vals(genotype, flavin, time):
    mask = ((df["Genotype"] == genotype) &
            (df["Flavin"]   == flavin)   &
            (df["Incubation_time"] == time))
    return df.loc[mask, "Net_uptake"].dropna().values


def welch(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return stats.ttest_ind(a, b, equal_var=False)


def sig_label(p):
    if np.isnan(p): return "n.d."
    if p >= 0.05:   return "ns"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"


# Planned comparisons: Col-0 vs bou at each timepoint, and
# the 10-vs-30 min kinetic effect within each genotype.
PLANNED = []
for flav in FLAVINS:
    PLANNED += [
        (flav, "Col-0", 10, "bou",   10, f"{flav}: Col-0 vs bou @ 10 min"),
        (flav, "Col-0", 30, "bou",   30, f"{flav}: Col-0 vs bou @ 30 min"),
        (flav, "Col-0", 10, "Col-0", 30, f"{flav}: Col-0 10 vs 30 min"),
        (flav, "bou",   10, "bou",   30, f"{flav}: bou 10 vs 30 min"),
    ]

N_COMP = len(PLANNED)
stat_records = []
for row in PLANNED:
    if len(row) == 6:
        f1, g1, t1, g2, t2, lbl = row
        f2 = f1
    else:
        f1, g1, t1, f2, g2, t2, lbl = row
    a = get_vals(g1, f1, t1)
    b = get_vals(g2, f2, t2)
    _, p = welch(a, b)
    p_bonf = min(p * N_COMP, 1.0) if not np.isnan(p) else np.nan
    stat_records.append({
        "Comparison": lbl,
        "Group1": f"{g1} {f1} {t1}min", "n1": len(a),
        "Mean1":  np.mean(a) if len(a) else np.nan,
        "Group2": f"{g2} {f2} {t2}min", "n2": len(b),
        "Mean2":  np.mean(b) if len(b) else np.nan,
        "p_raw":  p, "p_Bonferroni": p_bonf,
        "Sig":    sig_label(p_bonf),
    })

stats_df = pd.DataFrame(stat_records)
stats_df.to_csv("Flavin_Stats.csv", index=False)
print("\nStatistics:")
print(stats_df[["Comparison", "Mean1", "Mean2", "p_Bonferroni", "Sig"]].to_string())


def get_sig(flav, g1, t1, g2=None, t2=None, f2=None):
    """Look up the significance label for a comparison, either direction."""
    if g2 is None: g2 = g1
    if f2 is None: f2 = flav
    if t2 is None: t2 = t1
    for r in stat_records:
        if (r["Group1"] == f"{g1} {flav} {t1}min" and
                r["Group2"] == f"{g2} {f2} {t2}min"):
            return r["Sig"]
        if (r["Group1"] == f"{g2} {f2} {t2}min" and
                r["Group2"] == f"{g1} {flav} {t1}min"):
            return r["Sig"]
    return "n.d."


# Transport rate per replicate: (Net_uptake@30min - Net_uptake@10min) / 20 min
rate_records = []
for geno in GENOTYPES:
    for flav in FLAVINS:
        v10 = get_vals(geno, flav, 10)
        v30 = get_vals(geno, flav, 30)
        n   = min(len(v10), len(v30))
        if n == 0:
            continue
        rates = (v30[:n] - v10[:n]) / 20.0   # pmol.mg-1.min-1
        rate_records.append({
            "Genotype": geno, "Flavin": flav,
            "Rate_mean": np.mean(rates),
            "Rate_sd":   np.std(rates, ddof=1),
            "Rates":     rates,
        })
rates_df = pd.DataFrame(rate_records)

per_replicate_rows = []
for row in rate_records:
    for rep_idx, r in enumerate(row["Rates"], start=1):
        per_replicate_rows.append({
            "Genotype": row["Genotype"],
            "Flavin":   row["Flavin"],
            "Replicate": rep_idx,
            "Uptake_rate_pmol_mg_min": r,
        })
rates_per_replicate_df = pd.DataFrame(per_replicate_rows)
rates_per_replicate_df.to_csv("Flavin_Transport_Rate_per_replicate.csv", index=False)

rates_summary_df = rates_df[["Genotype", "Flavin", "Rate_mean", "Rate_sd"]].rename(
    columns={"Rate_mean": "Uptake_rate_mean_pmol_mg_min",
             "Rate_sd":   "Uptake_rate_sd_pmol_mg_min"})
rates_summary_df.to_csv("Flavin_Transport_Rate_summary.csv", index=False)

print("\nSaved transport-rate data:")
print(f"  Flavin_Transport_Rate_per_replicate.csv ({len(rates_per_replicate_df)} rows)")
print(f"  Flavin_Transport_Rate_summary.csv ({len(rates_summary_df)} rows)")


def draw_bracket(ax, x1, x2, y, label, color="black", lw=1.0, fontsize=None):
    """Draw a significance bracket between x1 and x2 at height y."""
    if label == "ns":
        return
    if fontsize is None:
        fontsize = FONT_ANNOT
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=lw)
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color=color)


# Figure 1 -- kinetic uptake line plot, FAD only
fig1, ax_single = plt.subplots(1, 1, figsize=(6.5, 6.2))
fig1.patch.set_facecolor("white")
axes = [ax_single]

colors  = {"Col-0": COL0_COLOR, "bou": BOU_COLOR}
fills   = {"Col-0": COL0_FILL,  "bou": BOU_FILL}
markers = {"Col-0": "o", "bou": "s"}

for ax, flav, panel_lbl in zip(axes, FLAVINS, [""]):
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

    for geno in GENOTYPES:
        col = colors[geno]
        fil = fills[geno]
        mk  = markers[geno]

        means, sds, xs = [], [], []
        for tp in TIMEPOINTS:
            v = get_vals(geno, flav, tp)
            means.append(np.mean(v))
            sds.append(np.std(v, ddof=1))
            xs.append(tp)

            jitter = np.random.uniform(-0.6, 0.6, len(v))
            ax.scatter(tp + jitter, v, color=fil, edgecolors=col,
                       s=JITTER_SIZE, zorder=3, alpha=0.75, linewidth=1.1)

        means, sds = np.array(means), np.array(sds)

        ax.fill_between(xs, means - sds, means + sds, color=col, alpha=0.12, zorder=1)
        ax.plot(xs, means, color=col, lw=2.4, marker=mk, markersize=8,
                markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.2,
                zorder=4, label=geno, solid_capstyle="round")
        ax.errorbar(xs, means, yerr=sds, fmt="none", color=col,
                    capsize=4, capthick=1.4, linewidth=1.4, zorder=5)

    ax.set_xlim(6, 34)
    ax.set_xticks(TIMEPOINTS)
    ax.set_xticklabels([f"{t} min" for t in TIMEPOINTS], fontsize=FONT_TICK, fontweight="bold")
    ax.set_xlabel("Incubation time (min)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Net uptake (pmol · mg⁻¹ protein)", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 3))
    ax.set_title(f"{flav}", fontsize=FONT_TITLE, fontweight="bold")
    ax.text(-0.10, 1.04, panel_lbl, transform=ax.transAxes, fontsize=16, fontweight="bold")

leg_els = [
    Line2D([0], [0], color=COL0_COLOR, lw=2.4, marker="o", markersize=8,
           markerfacecolor=COL0_COLOR, markeredgecolor="white", label="Col-0 (WT)"),
    Line2D([0], [0], color=BOU_COLOR, lw=2.4, marker="s", markersize=8,
           markerfacecolor=BOU_COLOR, markeredgecolor="white", label="bou"),
    Patch(facecolor=COL0_FILL, edgecolor="none", alpha=0.4, label="± SD (Col-0)"),
    Patch(facecolor=BOU_FILL,  edgecolor="none", alpha=0.4, label="± SD (bou)"),
]
fig1.legend(handles=leg_els, fontsize=FONT_LEGEND, loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 1.01), frameon=False)

plt.suptitle("Mitochondrial net uptake of FAD",
             fontsize=FONT_TITLE + 0.5, fontweight="bold", y=1.08)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Flavin_Kinetic_Uptake.png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("Flavin_Kinetic_Uptake.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved: Flavin_Kinetic_Uptake.png / .pdf")
plt.show()
