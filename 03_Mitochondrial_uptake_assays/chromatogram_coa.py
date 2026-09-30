#!/usr/bin/env python3
"""
chromatogram_coa_standards.py
Standards verification chromatograms for ¹³C-labeled O-CoA and P-CoA.
5 µM standard, 2 replicates, 2 transitions each.
Same style as chromatogram_coa.py.

Sheets: OCOA_STD, PCOA_STD
  OCOA_STD columns: O391_R1_t, O391_R1_s, O391_R2_t, O391_R2_s,
                    O428_R1_t, O428_R1_s, O428_R2_t, O428_R2_s
  PCOA_STD columns: P500_R1_t, P500_R1_s, P500_R2_t, P500_R2_s,
                    P428_R1_t, P428_R1_s, P428_R2_t, P428_R2_s

Output:
  Chromatogram_OCoA_STD.png / .pdf   — O-CoA: both transitions overlaid
  Chromatogram_PCoA_STD.png / .pdf   — P-CoA: both transitions overlaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ── Configuration (identical to chromatogram_coa.py) ──────────────────
EXCEL_FILE       = "Chromatogram.xlsx"
SMOOTHING_WINDOW = 21
SMOOTHING_POLY   = 3
INTERP_POINTS    = 2000
DPI              = 2500
FONT_TITLE       = 16
FONT_LABEL       = 14.5
FONT_TICK        = 14.5
FONT_ANNOT       = 13.5

# Colors — same as sample chromatograms
# Quantitative transition: darker shade
# Qualitative transition:  lighter shade of same color
OCOA_QUANT_COL   = "#E8837E"   # 898→391 qualitative
OCOA_QUANT_SD    = "#F5C5C3"
OCOA_QUAL_COL    = "#C03830"   # 898→428 quantitative
OCOA_QUAL_SD     = "#F0A09D"

PCOA_QUANT_COL   = "#4e4e52"   # 1007→500 quantitative
PCOA_QUANT_SD    = "#B0B0B4"
PCOA_QUAL_COL    = "#8e8e92"   # 1007→428 qualitative
PCOA_QUAL_SD     = "#CECECE"

trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ── Load two transitions from one STD sheet ───────────────────────────
def load_std_transition(df, t_cols, s_cols, rt_window):
    """
    Load one transition (2 replicates) from a standard sheet.
    t_cols, s_cols: lists of column names for time and signal replicates.
    Returns same dict structure as load_sheet in sample script.
    """
    t_min, t_max = rt_window
    t_grid = np.linspace(t_min, t_max, INTERP_POINTS)

    interp_signals = []
    aucs            = []
    raw_cropped     = []

    for tc_col, sc_col in zip(t_cols, s_cols):
        if tc_col not in df.columns or sc_col not in df.columns:
            print(f"  WARNING: columns {tc_col}/{sc_col} not found — skipping")
            continue
        t = df[tc_col].dropna().values
        s = df[sc_col].dropna().values
        n = min(len(t), len(s))
        t, s = t[:n], s[:n]

        mask = (t >= t_min) & (t <= t_max)
        tc, sc = t[mask], s[mask]
        if len(tc) < 5:
            continue

        wl = min(SMOOTHING_WINDOW,
                 len(sc) if len(sc) % 2 == 1 else len(sc) - 1)
        wl = max(wl, 5)
        sc_smooth = savgol_filter(sc, wl, SMOOTHING_POLY)
        sc_smooth = np.maximum(sc_smooth, 0)

        f_interp = interp1d(tc, sc_smooth, bounds_error=False,
                            fill_value=0.0, kind="linear")
        interp_signals.append(f_interp(t_grid))
        aucs.append(trapz_fn(sc_smooth, tc))
        raw_cropped.append((tc, sc_smooth))

    if not interp_signals:
        return None

    interp_signals = np.array(interp_signals)
    mean_sig = interp_signals.mean(axis=0)
    sd_sig   = interp_signals.std(axis=0, ddof=1) if len(interp_signals) > 1 \
               else np.zeros_like(mean_sig)

    # Second smoothing pass
    mean_sig = savgol_filter(mean_sig, 51, 3)
    mean_sig = np.maximum(mean_sig, 0)
    sd_sig   = savgol_filter(sd_sig, 51, 3)
    sd_sig   = np.maximum(sd_sig, 0)

    apex_idx = np.argmax(mean_sig)
    return {
        "t_grid":   t_grid,
        "mean":     mean_sig,
        "sd":       sd_sig,
        "traces":   raw_cropped,
        "apex_t":   t_grid[apex_idx],
        "apex_s":   mean_sig[apex_idx],
        "auc_mean": np.mean(aucs),
        "auc_sd":   np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
        "n":        len(interp_signals),
    }


# ── Draw one transition on an axis ────────────────────────────────────
def draw_transition(ax, d, color, color_sd, label,
                    rt_window, show_individual=True,
                    show_apex=True):
    if d is None:
        return

    if show_individual:
        for tc, sc in d["traces"]:
            ax.plot(tc, sc, color=color, lw=0.6, alpha=0.22)

    ax.fill_between(d["t_grid"],
                    np.maximum(d["mean"] - d["sd"], 0),
                    d["mean"] + d["sd"],
                    color=color_sd, alpha=0.45, linewidth=0)

    ax.plot(d["t_grid"], d["mean"],
            color=color, lw=2.2, label=label, zorder=4)

    if show_apex:
        ax.axvline(d["apex_t"], color=color, lw=0.9,
                   ls="--", alpha=0.55, zorder=3)
        ax.scatter(d["apex_t"], d["apex_s"],
                   color=color, s=55, zorder=6,
                   edgecolors="white", linewidth=1.0)
        ax.annotate(f"RT = {d['apex_t']:.2f} min",
                    xy=(d["apex_t"], d["apex_s"]),
                    xytext=(d["apex_t"] + 0.16,
                            d["apex_s"] * 0.78),
                    fontsize=FONT_ANNOT, color=color,
                    arrowprops=dict(arrowstyle="->",
                                   color=color, lw=0.9))


def style_ax(ax, rt_window, ylabel_show=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlim(rt_window)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Retention time (min)",
                  fontsize=FONT_LABEL, fontweight="bold")
    if ylabel_show:
        ax.set_ylabel("Signal intensity (a.u.)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_major_formatter(
        ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


# ── Load both STD sheets ───────────────────────────────────────────────
print("Loading standard chromatogram data...")

df_o = pd.read_excel(EXCEL_FILE, sheet_name="OCOA_STD")
df_o.columns = df_o.columns.str.strip()
df_p = pd.read_excel(EXCEL_FILE, sheet_name="PCOA_STD")
df_p.columns = df_p.columns.str.strip()

# O-CoA — define rt_window from expected peak region
O_RT = (6.0, 15.0)
P_RT = (5.0, 14.0)

o_quant = load_std_transition(df_o,
                               ["O428_R1_t", "O428_R2_t"],
                               ["O428_R1_s", "O428_R2_s"],
                               O_RT)
o_qual  = load_std_transition(df_o,
                               ["O391_R1_t", "O391_R2_t"],
                               ["O391_R1_s", "O391_R2_s"],
                               O_RT)

p_quant = load_std_transition(df_p,
                               ["P500_R1_t", "P500_R2_t"],
                               ["P500_R1_s", "P500_R2_s"],
                               P_RT)
p_qual  = load_std_transition(df_p,
                               ["P428_R1_t", "P428_R2_t"],
                               ["P428_R1_s", "P428_R2_s"],
                               P_RT)

print(f"  O-CoA quant (898→428): {'OK' if o_quant else 'MISSING'}")
print(f"  O-CoA qual  (898→391): {'OK' if o_qual  else 'MISSING'}")
print(f"  P-CoA quant (1007→500): {'OK' if p_quant else 'MISSING'}")
print(f"  P-CoA qual  (1007→428): {'OK' if p_qual  else 'MISSING'}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — ¹³C-Octanoyl-CoA standard (both transitions overlaid)
# ═══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig1.patch.set_facecolor("white")

draw_transition(ax1, o_quant, OCOA_QUANT_COL, OCOA_QUANT_SD,
                "m/z 898 → 428 (quantitative)", O_RT)
draw_transition(ax1, o_qual,  OCOA_QUAL_COL,  OCOA_QUAL_SD,
                "m/z 898 → 391 (qualitative)",  O_RT)

style_ax(ax1, O_RT)

ax1.set_title("Synthesized ¹³C-Octanoyl-CoA (4 µM)",
              fontsize=FONT_TITLE, fontweight="bold", pad=10)

# Legend
leg_els_o = [
        Line2D([0],[0], color=OCOA_QUAL_COL, lw=2.2,
           label="m/z 898.2 → 391.2"),
        Line2D([0],[0], color=OCOA_QUANT_COL, lw=2.2,
           label="m/z 898.2 → 428"),
    Patch(facecolor=OCOA_QUAL_SD, edgecolor="none",
          alpha=0.5, label="± SD"),
]
ax1.legend(handles=leg_els_o, fontsize=FONT_ANNOT,
           loc="upper right", frameon=False)

ax1.text(0.02, 0.97, "Synthesis\nverification",
         transform=ax1.transAxes, ha="left", va="top",
         fontsize=FONT_ANNOT, color="gray", style="italic")

plt.tight_layout()
plt.savefig("Chromatogram_OCoA_STD.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Chromatogram_OCoA_STD.pdf",
            bbox_inches="tight", facecolor="white")
print("\nSaved: Chromatogram_OCoA_STD.png / .pdf")
plt.show()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — ¹³C-Palmitoyl-CoA standard (both transitions overlaid)
# ═══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.patch.set_facecolor("white")

draw_transition(ax2, p_quant, PCOA_QUANT_COL, PCOA_QUANT_SD,
                "m/z 1007 → 500 (quantitative)", P_RT)
draw_transition(ax2, p_qual,  PCOA_QUAL_COL,  PCOA_QUAL_SD,
                "m/z 1007 → 428 (qualitative)",  P_RT)

style_ax(ax2, P_RT)

ax2.set_title("Synthesized ¹³C-Palmitoyl-CoA (4 µM)",
              fontsize=FONT_TITLE, fontweight="bold", pad=10)

leg_els_p = [
    Line2D([0],[0], color=PCOA_QUANT_COL, lw=2.2,
           label="m/z 1007.3 → 500.3"),
    Line2D([0],[0], color=PCOA_QUAL_COL, lw=2.2,
           label="m/z 1007.3 → 428"),
    Patch(facecolor=PCOA_QUANT_SD, edgecolor="none",
          alpha=0.5, label="± SD"),
]
ax2.legend(handles=leg_els_p, fontsize=FONT_ANNOT,
           loc="upper right", frameon=False)

ax2.text(0.02, 0.97, "Synthesis\nverification",
         transform=ax2.transAxes, ha="left", va="top",
         fontsize=FONT_ANNOT, color="gray", style="italic")

plt.tight_layout()
plt.savefig("Chromatogram_PCoA_STD.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.savefig("Chromatogram_PCoA_STD.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: Chromatogram_PCoA_STD.png / .pdf")
plt.show()

print("\nStandard chromatogram figures complete.")
