#!/usr/bin/env python3
"""
Publication chromatogram figure for labeled acylcarnitines (Col-0 only;
bou is background noise, not shown). Each compound gets one panel with
its quantitative (->85 Da) and qualitative (->60 Da) transitions
overlaid on the same axis -- darker shade = quantitative, lighter
shade of the same color = qualitative.

Sheets: O85, P85, O60, P60
Columns: Col0_R1_t, Col0_R1_s, Col0_R2_t, Col0_R2_s, ...

Output:
  Chromatogram_OCarnitine.png / .pdf
  Chromatogram_PCarnitine.png / .pdf
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

EXCEL_FILE       = "Chromatogram.xlsx"
SMOOTHING_WINDOW = 21   # Savitzky-Golay window (must be odd)
SMOOTHING_POLY   = 3
INTERP_POINTS    = 2000
DPI              = 2500
FONT_TITLE       = 16
FONT_LABEL       = 14.5
FONT_TICK        = 13
FONT_ANNOT       = 12

OCAR_QUANT_COL = "#C03830"   # 292 -> 85, quantitative
OCAR_QUANT_SD  = "#F0A09D"
OCAR_QUAL_COL  = "#E8837E"   # 292 -> 60, qualitative (lighter red)
OCAR_QUAL_SD   = "#F5C5C3"

PCAR_QUANT_COL = "#4e4e52"   # 401 -> 85, quantitative
PCAR_QUANT_SD  = "#B0B0B4"
PCAR_QUAL_COL  = "#8e8e92"   # 401 -> 60, qualitative (lighter grey)
PCAR_QUAL_SD   = "#CECECE"

O_RT = (3.5, 5.3)   # Octanoyl-carnitine retention-time window (min)
P_RT = (1.9, 4.6)   # Palmitoyl-carnitine retention-time window (min)

trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_sheet(sheet_name, rt_window):
    """
    Load one sheet (Col-0 replicates) and return a dict with:
      t_grid, mean, sd, traces (raw per-replicate), apex_t, apex_s,
      auc_mean, auc_sd, n
    """
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    reps = []
    r_idx = 1
    while True:
        t_col = f"Col0_R{r_idx}_t"
        s_col = f"Col0_R{r_idx}_s"
        if t_col not in df.columns:
            break
        t = df[t_col].dropna().values
        s = df[s_col].dropna().values
        n = min(len(t), len(s))
        reps.append((t[:n], s[:n]))
        r_idx += 1

    print(f"  {sheet_name}: {len(reps)} replicates loaded")

    t_min, t_max = rt_window
    t_grid = np.linspace(t_min, t_max, INTERP_POINTS)

    interp_signals = []
    aucs = []
    raw_cropped = []
    for t, s in reps:
        mask = (t >= t_min) & (t <= t_max)
        tc, sc = t[mask], s[mask]
        if len(tc) < 5:
            continue
        wl = min(SMOOTHING_WINDOW, len(sc) if len(sc) % 2 == 1 else len(sc) - 1)
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


def draw_transition(ax, d, color, color_sd, label, show_individual=True,
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

    ax.plot(d["t_grid"], d["mean"], color=color, lw=2.2, label=label, zorder=4)

    if show_apex:
        ax.axvline(d["apex_t"], color=color, lw=0.9, ls="--", alpha=0.55, zorder=3)
        ax.scatter(d["apex_t"], d["apex_s"], color=color, s=55, zorder=6,
                   edgecolors="white", linewidth=1.0)
        ax.annotate(f"RT = {d['apex_t']:.2f} min",
                    xy=(d["apex_t"], d["apex_s"]),
                    xytext=(d["apex_t"] + 0.06, d["apex_s"] * 0.88),
                    fontsize=FONT_ANNOT, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))


def style_ax(ax, rt_window, ylabel_show=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlim(rt_window)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Retention time (min)", fontsize=FONT_LABEL, fontweight="bold")
    if ylabel_show:
        ax.set_ylabel("Signal intensity (a.u.)", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


print("Loading chromatogram data...")
o_quant = load_sheet("O85", O_RT)   # 292 -> 85, quantitative
o_qual  = load_sheet("O60", O_RT)   # 292 -> 60, qualitative
p_quant = load_sheet("P85", P_RT)   # 401 -> 85, quantitative
p_qual  = load_sheet("P60", P_RT)   # 401 -> 60, qualitative

print(f"  O-carnitine quant (292->85): {'OK' if o_quant else 'MISSING'}")
print(f"  O-carnitine qual  (292->60): {'OK' if o_qual  else 'MISSING'}")
print(f"  P-carnitine quant (401->85): {'OK' if p_quant else 'MISSING'}")
print(f"  P-carnitine qual  (401->60): {'OK' if p_qual  else 'MISSING'}")


# Figure 1 -- Octanoyl-carnitine, both transitions overlaid
fig1, ax1 = plt.subplots(figsize=(7, 5))
fig1.patch.set_facecolor("white")

draw_transition(ax1, o_quant, OCAR_QUANT_COL, OCAR_QUANT_SD,
                "m/z 292 → 85 (quantitative)")
draw_transition(ax1, o_qual,  OCAR_QUAL_COL,  OCAR_QUAL_SD,
                "m/z 292 → 60 (qualitative)")

style_ax(ax1, O_RT)
ax1.set_title("¹³C-Octanoyl-carnitine", fontsize=FONT_TITLE, fontweight="bold", pad=10)

leg_els_o = [
    Line2D([0], [0], color=OCAR_QUANT_COL, lw=2.2, label="m/z 292.2 → 85.1"),
    Line2D([0], [0], color=OCAR_QUAL_COL, lw=2.2, label="m/z 292.2 → 60.2"),
    Patch(facecolor=OCAR_QUANT_SD, edgecolor="none", alpha=0.5, label="± SD"),
]
ax1.legend(handles=leg_els_o, fontsize=FONT_ANNOT, loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig("Chromatogram_OCarnitine.png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("Chromatogram_OCarnitine.pdf", bbox_inches="tight", facecolor="white")
print("\nSaved: Chromatogram_OCarnitine.png / .pdf")
plt.show()


# Figure 2 -- Palmitoyl-carnitine, both transitions overlaid
fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.patch.set_facecolor("white")

draw_transition(ax2, p_quant, PCAR_QUANT_COL, PCAR_QUANT_SD,
                "m/z 401 → 85 (quantitative)")
draw_transition(ax2, p_qual,  PCAR_QUAL_COL,  PCAR_QUAL_SD,
                "m/z 401 → 60 (qualitative)")

style_ax(ax2, P_RT)
ax2.set_title("¹³C-Palmitoyl-carnitine", fontsize=FONT_TITLE, fontweight="bold", pad=10)

leg_els_p = [
    Line2D([0], [0], color=PCAR_QUANT_COL, lw=2.2, label="m/z 401.3 → 85.1"),
    Line2D([0], [0], color=PCAR_QUAL_COL, lw=2.2, label="m/z 401.3 → 60.2"),
    Patch(facecolor=PCAR_QUANT_SD, edgecolor="none", alpha=0.5, label="± SD"),
]
ax2.legend(handles=leg_els_p, fontsize=FONT_ANNOT, loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig("Chromatogram_PCarnitine.png", dpi=DPI, bbox_inches="tight", facecolor="white")
plt.savefig("Chromatogram_PCarnitine.pdf", bbox_inches="tight", facecolor="white")
print("Saved: Chromatogram_PCarnitine.png / .pdf")
plt.show()
