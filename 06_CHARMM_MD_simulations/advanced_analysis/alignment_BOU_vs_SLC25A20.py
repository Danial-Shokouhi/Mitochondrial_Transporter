#!/usr/bin/env python3
"""
Figure 5 Panel E — Sequence alignment: BOU vs SLC25A20 (human)
Highlights conservation at gate and binding-site positions.
Annotates ARG296/ASP7 (plant-specific c-gate) vs GLY/MET at 
equivalent human positions (uncharged).

>sp|Q93XM7|MCAT_ARATH Mitochondrial carnitine/acylcarnitine carrier-like protein OS=Arabidopsis thaliana OX=3702 GN=BOU PE=1 SV=1
MADAWKDLASGTVGGAAQLVVGHPFDTIKVKLQSQPTPAPGQLPRYTGAIDAVKQTVASE
GTKGLYKGMGAPLATVAAFNAVLFTVRGQMEGLLRSEAGVPLTISQQFVAGAGAGFAVSF
LACPTELIKCRLQAQGALAGASTTSSVVAAVKYGGPMDVARHVLRSEGGARGLFKGLFPT
FAREVPGNATMFAAYEAFKRFLAGGSDTSSLGQGSLIMAGGVAGASFWGIVYPTDVVKSV
LQVDDYKNPRYTGSMDAFRKILKSEGVKGLYKGFGPAMARSVPANAACFLAYEMTRSSLG

>sp|O43772|MCAT_HUMAN Mitochondrial carnitine/acylcarnitine carrier protein OS=Homo sapiens OX=9606 GN=SLC25A20 PE=1 SV=1
MADQPKPISPLKNLLAGGFGGVCLVFVGHPLDTVKVRLQTQPPSLPGQPPMYSGTFDCFR
KTLFREGITGLYRGMAAPIIGVTPMFAVCFFGFGLGKKLQQKHPEDVLSYPQLFAAGMLS
GVFTTGIMTPGERIKCLLQIQASSGESKYTGTLDCAKKLYQEFGIRGIYKGTVLTLMRDV
PASGMYFMTYEWLKNIFTPEGKRVSELSAPRILVAGGIAGIFNWAVAIPPDVLKSRFQTA
PPGKYPNGFRDVLRELIRDEGVTSLYKGFNAVMIRAFPANAACFLGFEVAMKFLNWATPN
L

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Rectangle
import re, sys, textwrap

try:
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    from Bio import SeqIO
    from io import StringIO
except ImportError:
    sys.exit("Install Biopython: conda install -c conda-forge biopython")

# SEQUENCES (read from file or hardcoded below)

SEQ_FILE = ".../rep1"

def read_fasta(path):
    seqs = {}
    with open(path) as f:
        content = f.read()
    for block in content.strip().split(">"):
        if not block.strip(): continue
        lines = block.strip().splitlines()
        header = lines[0]
        seq    = "".join(l.strip() for l in lines[1:])
        tag    = "BOU" if "ARATH" in header or "BOU" in header else \
                 "SLC25A20" if "HUMAN" in header or "SLC25A20" in header else \
                 header.split()[0][:10]
        seqs[tag] = seq
    return seqs

seqs = read_fasta(SEQ_FILE)
BOU_SEQ     = seqs.get("BOU",     "")
HUMAN_SEQ   = seqs.get("SLC25A20","")

if not BOU_SEQ or not HUMAN_SEQ:
    sys.exit(f"Could not parse sequences. Found keys: {list(seqs.keys())}")

print(f"BOU:      {len(BOU_SEQ)} aa")
print(f"SLC25A20: {len(HUMAN_SEQ)} aa")

# KEY RESIDUE ANNOTATIONS (BOU 1-based positions)

# c-gate residues (BOU positions, 1-based)
CGATE_BOU = {7: "ASP7", 87: "ARG87", 196: "GLU196",
              199: "LYS199", 293: "GLU293", 296: "ARG296"}

# m-gate residues
MGATE_BOU = {26: "ASP26", 29: "LYS29", 126: "GLU126",
              129: "LYS129", 235: "ASP235", 238: "LYS238"}

# Binding-site residues
BIND_BOU = {18: "GLN18", 183: "ARG183", 184: "GLU184",
            228: "TRP228", 280: "ARG280"}

# Plant-specific novel c-gate pair (highlight with star)
NOVEL_BOU = {7: "ASP7", 296: "ARG296"}

# Human SLC25A20 equivalents (human 1-based positions)
# Box-framed only, same colors as the BOU side, no text labels, so
# the figure doesn't get crowded with a second tier of residue names.
BIND_HUMAN  = {29: "HIS29", 178: "ARG178", 179: "ASP179",
               224: "TRP224", 275: "ARG275"}
MGATE_HUMAN = {32: "ASP32", 35: "LYS35", 132: "GLU132",
               135: "LYS135", 231: "ASP231", 234: "LYS234"}
CGATE_HUMAN = {97: "LYS97", 94: "GLY94", 191: "GLU191",
               194: "LYS194", 288: "GLU288", 291: "MET291"}

# Color scheme
COL_CGATE  = "#E74C3C"    # red
COL_MGATE  = "#2980B9"    # blue
COL_BIND   = "#F39C12"    # gold
COL_NOVEL  = "#8E44AD"    # purple (plant-specific)
COL_CONS   = "#27AE60"    # green (conserved)
COL_HUMAN  = "#95A5A6"    # grey (human unique)

# Residue physicochemical colours (single letter background)
AA_COLORS = {
    "R": "#1E90FF", "K": "#1E90FF", "H": "#63A0FF",   # positive
    "D": "#FF4500", "E": "#FF4500",                    # negative
    "F": "#FFD700", "Y": "#FFD700", "W": "#FFD700",   # aromatic
    "A": "#C8C8C8", "V": "#C8C8C8", "L": "#C8C8C8",
    "I": "#C8C8C8", "M": "#C8C8C8", "P": "#C8C8C8",   # aliphatic
    "G": "#EBEBEB",                                    # glycine
    "S": "#98FB98", "T": "#98FB98",                   # hydroxyl
    "N": "#98FB98", "Q": "#98FB98",                   # amide
    "C": "#FFDEAD",                                    # cys
    "-": "white",
}

# PAIRWISE ALIGNMENT

print("Running pairwise alignment...")
from Bio.Align import substitution_matrices
blosum62 = substitution_matrices.load('BLOSUM62')
try:
    alignments = pairwise2.align.globalds(
        BOU_SEQ, HUMAN_SEQ,
        blosum62, -10, -0.5)
    print('  Using BLOSUM62 alignment')
except Exception:
    alignments = pairwise2.align.globalms(
        BOU_SEQ, HUMAN_SEQ,
        2, -1, -8, -0.5)
    print('  Using simple match alignment')

aln     = alignments[0]
aln_bou, aln_human, score, begin, end = aln
aln_len = len(aln_bou)

print(f"Alignment length: {aln_len}")
print(f"Alignment score:  {score:.1f}")

# Map BOU alignment positions to sequence positions
bou_pos   = []   # actual BOU residue number at each alignment col (0 = gap)
human_pos = []
bpos = 0; hpos = 0
for i in range(aln_len):
    if aln_bou[i] != "-":
        bpos += 1
    if aln_human[i] != "-":
        hpos += 1
    bou_pos.append(bpos if aln_bou[i] != "-" else 0)
    human_pos.append(hpos if aln_human[i] != "-" else 0)

# Conservation string
cons_str = []
for b, h in zip(aln_bou, aln_human):
    if b == h and b != "-":
        cons_str.append("|")
    elif b != "-" and h != "-":
        cons_str.append(":")
    else:
        cons_str.append(" ")
cons_str = "".join(cons_str)

# Identity
identical = sum(1 for b,h in zip(aln_bou,aln_human)
                if b==h and b!="-")
total_non_gap = sum(1 for b in aln_bou if b!="-")
identity = 100.0 * identical / max(total_non_gap,1)
print(f"Identity: {identity:.1f}%")

# Find human positions equivalent to novel BOU positions (7, 296)
equiv_human = {}
for bou_r, lbl in NOVEL_BOU.items():
    for i, (bp, hp) in enumerate(zip(bou_pos, human_pos)):
        if bp == bou_r and hp > 0:
            equiv_human[bou_r] = (hp, aln_human[i])
            break
        elif bp == bou_r and hp == 0:
            # gap — find nearest
            equiv_human[bou_r] = (None, "-")
            break

print("\nNovel c-gate equivalences:")
for bou_r, (hr, haa) in equiv_human.items():
    print(f"  BOU {NOVEL_BOU[bou_r]} (pos {bou_r}) "
          f"↔ Human pos {hr} = {haa}")

# 
# FIGURE — publication alignment display
# Row layout: 60 residues per row (alignment columns)
# Each row: BOU line, conservation line, HUMAN line
# Key positions marked with colored boxes above/below
# 

COLS_PER_ROW = 60
CHAR_W       = 0.14   # inches per character
CHAR_H       = 0.20   # inches per character row
BLOCK_H      = 0.95   # total height per alignment block (3 rows + labels)
ANNO_H       = 0.68   # height for annotation above/below each block
                       # (room for a 2nd label tier)
LR_MARGIN    = 1.2    # left margin inches (for sequence labels)

n_blocks = int(np.ceil(aln_len / COLS_PER_ROW))
fig_w    = LR_MARGIN + COLS_PER_ROW * CHAR_W + 1.0
fig_h    = n_blocks * (BLOCK_H + ANNO_H) + 1.0

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.axis("off")
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)

# All residue sets for annotation lookup
all_bou_annot  = {**CGATE_BOU,  **MGATE_BOU,  **BIND_BOU}
annot_colors   = {}
for r in CGATE_BOU:  annot_colors[r] = COL_NOVEL  if r in NOVEL_BOU \
                                        else COL_CGATE
for r in MGATE_BOU:  annot_colors[r] = COL_MGATE
for r in BIND_BOU:
    if r not in annot_colors:
        annot_colors[r] = COL_BIND

# Human-side frame colors — same three colors, no labels/legend changes
human_annot_colors = {}
for r in CGATE_HUMAN: human_annot_colors[r] = COL_CGATE
for r in MGATE_HUMAN: human_annot_colors[r] = COL_MGATE
for r in BIND_HUMAN:
    if r not in human_annot_colors:
        human_annot_colors[r] = COL_BIND

def draw_alignment_block(block_idx, block_start, block_end):
    """Draw one block of the alignment."""
    y_base = fig_h - 1.0 - block_idx * (BLOCK_H + ANNO_H)

    col_range = range(block_start, min(block_end, aln_len))
    n_col     = len(col_range)

    bou_chunk   = aln_bou[block_start:block_end]
    hum_chunk   = aln_human[block_start:block_end]
    cons_chunk  = cons_str[block_start:block_end]
    bpos_chunk  = bou_pos[block_start:block_end]
    hpos_chunk  = human_pos[block_start:block_end]

    # Starting positions for labels
    bstart_lbl = next((p for p in bpos_chunk  if p > 0), "")
    hstart_lbl = next((p for p in hpos_chunk if p > 0), "")

    #  Row labels 
    ax.text(LR_MARGIN - 0.08, y_base,        "BOU ",
            ha="right", va="center", fontsize=7.5,
            fontweight="bold", color="#2C3E50")
    ax.text(LR_MARGIN - 0.08, y_base - CHAR_H*1.5, "     ",
            ha="right", va="center", fontsize=7, color="gray")
    ax.text(LR_MARGIN - 0.08, y_base - CHAR_H*3.0, "hCAC",
            ha="right", va="center", fontsize=7.5,
            fontweight="bold", color="#7F8C8D")

    # Position numbers sit further left than the row-tag text
    # ("BOU"/"hCAC") to avoid crowding
    ax.text(LR_MARGIN - 0.58, y_base, str(bstart_lbl) if bstart_lbl else "",
            ha="right", va="center", fontsize=6.5, color="#555")
    ax.text(LR_MARGIN - 0.58, y_base - CHAR_H*3.0,
            str(hstart_lbl) if hstart_lbl else "",
            ha="right", va="center", fontsize=6.5, color="#555")
    bend_lbl = next((p for p in reversed(bpos_chunk) if p > 0), "")
    hend_lbl = next((p for p in reversed(hpos_chunk) if p > 0), "")
    ax.text(LR_MARGIN + n_col*CHAR_W + 0.05, y_base,
            str(bend_lbl), ha="left", va="center",
            fontsize=6.5, color="#555")
    ax.text(LR_MARGIN + n_col*CHAR_W + 0.05, y_base - CHAR_H*3.0,
            str(hend_lbl), ha="left", va="center",
            fontsize=6.5, color="#555")

    #  Draw each character 
    for ci, (b, h, c, bp, hp) in enumerate(
            zip(bou_chunk, hum_chunk, cons_chunk,
                bpos_chunk, hpos_chunk)):
        xpos = LR_MARGIN + ci * CHAR_W

        # Background box for BOU residue
        bg_bou = AA_COLORS.get(b, "white")
        if b != "-":
            rect_b = Rectangle((xpos, y_base - CHAR_H*0.45),
                                CHAR_W*0.95, CHAR_H*0.9,
                                facecolor=bg_bou, edgecolor="none",
                                alpha=0.6, zorder=1)
            ax.add_patch(rect_b)

        # Annotation overlay for key BOU residues
        if bp in annot_colors and b != "-":
            col_a = annot_colors[bp]
            rect_a = Rectangle((xpos, y_base - CHAR_H*0.48),
                                CHAR_W*0.95, CHAR_H*0.92,
                                facecolor="none",
                                edgecolor=col_a, linewidth=1.8,
                                zorder=3)
            ax.add_patch(rect_a)
            # Novel positions — add star above
            if bp in NOVEL_BOU:
                ax.text(xpos + CHAR_W*0.45, y_base + CHAR_H*0.65,
                        "★", ha="center", va="bottom",
                        fontsize=7, color=COL_NOVEL, zorder=5)

        # Background for HUMAN residue
        bg_hum = AA_COLORS.get(h, "white")
        if h != "-":
            rect_h = Rectangle((xpos, y_base - CHAR_H*3.0 - CHAR_H*0.45),
                                CHAR_W*0.95, CHAR_H*0.9,
                                facecolor=bg_hum, edgecolor="none",
                                alpha=0.6, zorder=1)
            ax.add_patch(rect_h)

        # Box frame for key HUMAN residues (same colors as BOU side,
        # no text label / not added to the annotation bar — box only)
        if hp in human_annot_colors and h != "-":
            col_ha = human_annot_colors[hp]
            rect_ha = Rectangle((xpos, y_base - CHAR_H*3.0 - CHAR_H*0.48),
                                 CHAR_W*0.95, CHAR_H*0.92,
                                 facecolor="none",
                                 edgecolor=col_ha, linewidth=1.8,
                                 zorder=3)
            ax.add_patch(rect_ha)

        # BOU character
        fw_bou = "bold" if bp in all_bou_annot else "normal"
        ax.text(xpos + CHAR_W*0.45, y_base, b,
                ha="center", va="center",
                fontsize=7, fontweight=fw_bou,
                fontfamily="monospace", color="#1A1A1A", zorder=4)

        # Conservation character
        cons_col = {"|": COL_CONS, ":": "#7DCEA0"}.get(c, "#CCCCCC")
        if c in ("|", ":"):
            ax.text(xpos + CHAR_W*0.45, y_base - CHAR_H*1.5,
                    c, ha="center", va="center",
                    fontsize=6, fontfamily="monospace",
                    color=cons_col, zorder=4)

        # HUMAN character
        ax.text(xpos + CHAR_W*0.45, y_base - CHAR_H*3.0,
                h, ha="center", va="center",
                fontsize=7, fontweight="normal",
                fontfamily="monospace", color="#555555", zorder=4)

        # Every 10th position gets a tick mark, placed close to the
        # sequence row so it doesn't collide with the annotation
        # labels above
        if bp > 0 and bp % 10 == 0:
            ax.text(xpos + CHAR_W*0.45, y_base + CHAR_H*0.62,
                    str(bp), ha="center", va="bottom",
                    fontsize=5.5, color="#AAAAAA")

    # Annotation bar above each block
    # Each distinct annotated residue gets its own label segment (a
    # run of contiguous columns with the same residue); segments break
    # whenever the residue identity changes. Any two labels landing
    # close enough to collide are staggered onto a second vertical
    # tier instead of overlapping.
    y_tiers = [y_base + CHAR_H*1.3, y_base + CHAR_H*2.0]
    segments = []
    seg_start = None; seg_end = None; seg_bp = None
    for ci, bp in enumerate(bpos_chunk):
        active = bp in annot_colors
        if active and bp == seg_bp:
            seg_end = ci
        elif active:
            if seg_bp is not None:
                segments.append((seg_start, seg_end, seg_bp))
            seg_start = ci; seg_end = ci; seg_bp = bp
        else:
            if seg_bp is not None:
                segments.append((seg_start, seg_end, seg_bp))
            seg_bp = None
    if seg_bp is not None:
        segments.append((seg_start, seg_end, seg_bp))

    tier_end = [-1e9, -1e9]   # rightmost x used so far, per tier
    for seg_start, seg_end, bp in segments:
        x0 = LR_MARGIN + seg_start * CHAR_W
        x1 = LR_MARGIN + (seg_end + 1) * CHAR_W
        xc = (x0 + x1) / 2
        lbl_txt = all_bou_annot.get(bp, "")
        half_w  = len(lbl_txt) * 0.032 + 0.03   # rough label half-width
        tier = 1 if (xc - half_w < tier_end[0] + 0.03) else 0
        ax.text(xc, y_tiers[tier], lbl_txt,
                ha="center", va="bottom",
                fontsize=5.5, color=annot_colors[bp],
                fontweight="bold")
        tier_end[tier] = xc + half_w


# Draw all blocks
for bi in range(n_blocks):
    bs = bi * COLS_PER_ROW
    be = bs + COLS_PER_ROW
    draw_alignment_block(bi, bs, be)

#  Identity bar at bottom 
y_id = 0.34
ax.text(fig_w/2, y_id,
        f"Pairwise identity: {identity:.1f}%   |   "
        f"Alignment length: {aln_len} aa   |   "
        f"BOU: {len(BOU_SEQ)} aa   |   hCAC (SLC25A20): {len(HUMAN_SEQ)} aa",
        ha="center", va="center", fontsize=8, color="#555555")

#  Legend 
legend_items = [
    mpatches.Patch(edgecolor=COL_NOVEL,  facecolor="none",
                   linewidth=1.8, label="Plant-specific c-gate (ARG296/ASP7) ★"),
    mpatches.Patch(edgecolor=COL_CGATE,  facecolor="none",
                   linewidth=1.8, label="C-gate residues"),
    mpatches.Patch(edgecolor=COL_MGATE,  facecolor="none",
                   linewidth=1.8, label="M-gate residues"),
    mpatches.Patch(edgecolor=COL_BIND,   facecolor="none",
                   linewidth=1.8, label="Binding-site residues"),
    mpatches.Patch(facecolor="#1E90FF",   alpha=0.6,
                   label="Positive (R/K/H)"),
    mpatches.Patch(facecolor="#FF4500",   alpha=0.6,
                   label="Negative (D/E)"),
    mpatches.Patch(facecolor="#FFD700",   alpha=0.6,
                   label="Aromatic (F/Y/W)"),
    mpatches.Patch(facecolor="#98FB98",   alpha=0.6,
                   label="Polar (S/T/N/Q)"),
    mpatches.Patch(facecolor="#C8C8C8",   alpha=0.6,
                   label="Aliphatic"),
]
leg = ax.legend(handles=legend_items, fontsize=7,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=5, frameon=True,
                framealpha=0.92, edgecolor="#CCCCCC",
                bbox_transform=fig.transFigure)

#  Title 
ax.text(fig_w/2, fig_h - 0.35,
        r"Sequence Alignment: BOU vs SLC25A20 ($\mathit{Homo\ sapiens}$)",
        ha="center", va="top", fontsize=12, fontweight="bold",
        color="black")

plt.savefig("Fig5E_alignment.png", dpi=1000,
            bbox_inches="tight", facecolor="white")
plt.savefig("Fig5E_alignment.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: Fig5E_alignment.png / .pdf")
plt.show()
