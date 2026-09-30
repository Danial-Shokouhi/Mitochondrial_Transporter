#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

EXCEL_FILE = "Marker.xlsx"
SHEET_NAME = "analysis"

MARKERS = {
    "CS":          {"label": "Citrate Synthase\n(CS)",
                    "unit":  "µmol TNB min⁻¹ mg⁻¹",
                    "type":  "mitochondrial",
                    "expect": "enriched"},
    "HPR":         {"label": "Hydroxypyruvate\nReductase (HPR)",
                    "unit":  "nmol NAD min⁻¹ mg⁻¹",
                    "type":  "peroxisomal",
                    "expect": "depleted"},
    "UGPase":      {"label": "UDP-Glucose\nPyrophosphorylase\n(UGPase)",
                    "unit":  "nmol NADPH min⁻¹ mg⁻¹",
                    "type":  "cytosolic",
                    "expect": "depleted"},
    "Chlorophyll": {"label": "Total Chlorophyll",
                    "unit":  "µg mg⁻¹",
                    "type":  "plastidial",
                    "expect": "depleted"},
}

GENOTYPE_COLORS = {
    "Col-0": {"mito": "black", "crude": "#707071"},
    "bou":   {"mito": "#B22222", "crude": "#E8A0A0"},
}

GENOTYPES = ["Col-0", "bou"]

BREAK_CONFIG = {
    "HPR":         {"bottom_max": 23, "top_min": 340},
    "UGPase":      {"bottom_max": 33, "top_min": 1440},
    "Chlorophyll": {"bottom_max": 3.2,  "top_min": 42},
}

print(f"Loading {EXCEL_FILE}...")
df = pd.read_excel(EXCEL_FILE)
df.columns = df.columns.str.strip()
df["Fraction"] = df["Fraction"].str.strip().str.lower()
df["Genotype"] = df["Genotype"].str.strip()

print(f"  Rows: {len(df)}")
print(f"  Fractions: {df['Fraction'].unique()}")
print(f"  Genotypes: {df['Genotype'].unique()}")
print(df.head())

mito  = df[df["Fraction"] == "mito"].copy()
crude = df[df["Fraction"] == "crude"].copy()

# Enrichment ratio per replicate, matched by position within each genotype
enrichment_records = []
for geno in GENOTYPES:
    m = mito[mito["Genotype"] == geno].reset_index(drop=True)
    c = crude[crude["Genotype"] == geno].reset_index(drop=True)
    n = min(len(m), len(c))
    for i in range(n):
        rec = {"Genotype": geno, "Replicate": i + 1}
        for marker in MARKERS:
            m_val = m.loc[i, marker] if marker in m.columns else np.nan
            c_val = c.loc[i, marker] if marker in c.columns else np.nan
            rec[f"{marker}_mito"]  = m_val
            rec[f"{marker}_crude"] = c_val
            rec[f"{marker}_ratio"] = m_val / c_val if (c_val and c_val != 0) else np.nan
        enrichment_records.append(rec)

er = pd.DataFrame(enrichment_records)

print("\n" + "=" * 65)
print("ENRICHMENT RATIOS (Mito / Crude) — mean ± SD")
print("=" * 65)
for geno in GENOTYPES:
    g = er[er["Genotype"] == geno]
    print(f"\n  {geno}:")
    for marker in MARKERS:
        col = f"{marker}_ratio"
        if col in g.columns:
            vals = g[col].dropna()
            print(f"    {marker:12}: {vals.mean():.2f} ± {vals.std():.2f}  (n={len(vals)})")

er.to_csv("Mito_Purity_Stats.csv", index=False)
print("\nSaved: Mito_Purity_Stats.csv")

print("\nOne-sample t-test: enrichment ratio vs 1.0 (H0: no enrichment)")
for geno in GENOTYPES:
    g = er[er["Genotype"] == geno]
    for marker in MARKERS:
        col = f"{marker}_ratio"
        vals = g[col].dropna().values
        if len(vals) >= 3:
            t, p = stats.ttest_1samp(vals, 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {geno} {marker}: ratio={vals.mean():.2f}, p={p:.4f} {sig}")


fig = plt.figure(figsize=(16, 12))
gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1], hspace=0.45)

# Panel A: specific activities, Mito vs Crude, one subplot per marker
gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], wspace=0.45)

for col_idx, (marker, info) in enumerate(MARKERS.items()):
    brk = BREAK_CONFIG.get(marker)

    if brk is None:
        ax = fig.add_subplot(gs_top[col_idx])
        plot_axes = [ax]
    else:
        gs_cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_top[col_idx],
            height_ratios=[1, 1.3], hspace=0.08)
        ax_top    = fig.add_subplot(gs_cell[0])
        ax_bottom = fig.add_subplot(gs_cell[1])
        plot_axes = [ax_top, ax_bottom]

    for plot_ax in plot_axes:
        for g_idx, geno in enumerate(GENOTYPES):
            g = er[er["Genotype"] == geno]
            mito_vals  = g[f"{marker}_mito"].dropna().values
            crude_vals = g[f"{marker}_crude"].dropna().values

            x_mito  = g_idx * 2.0
            x_crude = g_idx * 2.0 + 0.75

            col_m = GENOTYPE_COLORS[geno]["mito"]
            col_c = GENOTYPE_COLORS[geno]["crude"]

            plot_ax.bar(x_mito, np.mean(mito_vals), 0.65,
                       color=col_m, alpha=0.85, edgecolor="white", lw=0.5,
                       label=f"{geno} Mito" if col_idx == 0 else "")
            plot_ax.bar(x_crude, np.mean(crude_vals), 0.65,
                       color=col_c, alpha=0.85, edgecolor="white", lw=0.5,
                       label=f"{geno} Crude" if col_idx == 0 else "")

            plot_ax.errorbar(x_mito, np.mean(mito_vals), yerr=np.std(mito_vals),
                            fmt="none", color="black", capsize=3, lw=1.2)
            plot_ax.errorbar(x_crude, np.mean(crude_vals), yerr=np.std(crude_vals),
                            fmt="none", color="black", capsize=3, lw=1.2)

            jitter_m = np.random.uniform(-0.12, 0.12, len(mito_vals))
            jitter_c = np.random.uniform(-0.12, 0.12, len(crude_vals))
            plot_ax.scatter(x_mito + jitter_m, mito_vals,
                           color="black", s=18, zorder=5, alpha=1.0)
            plot_ax.scatter(x_crude + jitter_c, crude_vals,
                           color="black", s=18, zorder=5, alpha=1.0)

    if brk is None:
        ax.set_xticks([0.375, 2.375])
        ax.set_xticklabels(GENOTYPES, fontsize=12, fontweight="bold")
        ax.set_ylabel(info["unit"], fontsize=12, fontweight="bold")
        ax.set_title(info["label"], fontsize=12, fontweight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax_top.set_ylim(brk["top_min"], None)
        ax_bottom.set_ylim(0, brk["bottom_max"])

        ax_top.set_xticks([])
        ax_top.tick_params(bottom=False)
        ax_bottom.set_xticks([0.375, 2.375])
        ax_bottom.set_xticklabels(GENOTYPES, fontsize=12, fontweight="bold")
        ax_bottom.set_ylabel(info["unit"], fontsize=12, fontweight="bold")
        ax_top.set_title(info["label"], fontsize=12, fontweight="bold", pad=6)

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.spines[["top", "right"]].set_visible(False)
        ax_bottom.spines[["top", "right"]].set_visible(False)

legend_els = []
for geno in GENOTYPES:
    legend_els.append(Patch(color=GENOTYPE_COLORS[geno]["mito"],
                            label=f"{geno} — Mitochondrial fraction"))
    legend_els.append(Patch(color=GENOTYPE_COLORS[geno]["crude"],
                            label=f"{geno} — Crude extract"))
fig.legend(handles=legend_els, loc="upper center",
           bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=12, frameon=False)

# Panel B: enrichment ratio per marker
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_main[1], wspace=0.35, width_ratios=[2, 1])

ax_ratio = fig.add_subplot(gs_bot[0])
marker_list = list(MARKERS.keys())

x_base = np.arange(len(marker_list))
width  = 0.32

for g_idx, geno in enumerate(GENOTYPES):
    g = er[er["Genotype"] == geno]
    ratio_means = []
    ratio_stds  = []
    for marker in marker_list:
        vals = g[f"{marker}_ratio"].dropna().values
        ratio_means.append(np.mean(vals))
        ratio_stds.append(np.std(vals) / np.sqrt(len(vals)))  # SEM

    x_pos = x_base + (g_idx - 0.5) * width
    col   = GENOTYPE_COLORS[geno]["mito"]
    ax_ratio.bar(x_pos, ratio_means, width, yerr=ratio_stds,
                color=col, alpha=0.85, edgecolor="white", lw=0.5,
                capsize=4, label=geno)

    for m_idx, marker in enumerate(marker_list):
        vals = g[f"{marker}_ratio"].dropna().values
        jitter = np.random.uniform(-0.06, 0.06, len(vals))
        ax_ratio.scatter(x_pos[m_idx] + jitter, vals,
                         color="black", s=18, zorder=5, alpha=0.7)

ax_ratio.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.6, label="Ratio = 1")
ax_ratio.set_xticks(x_base)
ax_ratio.set_xticklabels(
    ["CS\n(Mitochondrial)", "HPR\n(Peroxisomal)", "UGPase\n(Cytosolic)", "Chlorophyll\n(Plastidial)"],
    fontweight="bold", fontsize=13)
ax_ratio.set_ylabel("Enrichment Ratio", fontsize=13, fontweight="bold")
ax_ratio.legend(fontsize=12, frameon=False)
ax_ratio.spines[["top", "right"]].set_visible(False)

for i, (marker, info) in enumerate(MARKERS.items()):
    arrow = "▲ enriched" if info["expect"] == "enriched" else "▼ depleted"
    col   = "white" if info["expect"] == "enriched" else "#004080"
    ax_ratio.text(i, ax_ratio.get_ylim()[0], arrow, ha="center", va="bottom",
                  fontsize=9, color=col, fontweight="bold")

fig.text(0.01, 0.94, "A", fontsize=16, fontweight="bold")
fig.text(0.01, 0.44, "B", fontsize=16, fontweight="bold")

plt.suptitle(
    "Mitochondrial Isolation Quality and Purity\n"
    "Marker enzyme activities — Col-0 and bou mutant",
    fontsize=13, fontweight="bold", y=1.00)

plt.savefig("Mito_Purity.png", dpi=1000, bbox_inches="tight")
plt.savefig("Mito_Purity.pdf", bbox_inches="tight")
print("\nSaved: Mito_Purity.png / .pdf")
plt.show()
