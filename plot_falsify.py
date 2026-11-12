#!/usr/bin/env python
"""
plot_falsify.py
===============

THE FALSIFICATION FIGURE — the experiment built to kill the mechanism.

Every system measured before this one dissociated HOMOLYTICALLY, so the global
singlet label was false at large separation in all of them. Envelope-closing and
false-label were perfectly confounded through the choice of chemistry.

Two designs break that confound:

  TRUE-LABEL STRONG BONDS -- stabilised singlet carbenes. C2F4 -> 2 CF2 and
  C2Cl4 -> 2 CCl2 dissociate to genuine closed-shell singlets, so the "neutral
  singlet" label stays TRUE at every separation, at bond strengths of 78 and 119
  kcal/mol. If the ramp tracked bond strength, C2Cl4 -- the strongest bond in the
  study -- should ramp hardest.

  THE SPIN-ISOLATING TRIPLE -- panel (c). The carbenes show "all labels true ->
  no ramp", but they cannot say WHICH label, because a symmetric homolysis into
  two identical halves preserves elemental composition exactly. Ethane closes
  that gap: C2H6 -> 2 CH3 is the same symmetric homolysis with composition
  preserved identically (25% C / 75% H either way), but the fragments are
  RADICALS. Only the spin label changes, and the ramp changes by 100x.
"""
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# Publication form by default: no suptitle, no panel titles, no on-figure
# arrows, ratios or notes. --annotate restores the slide version. The caption
# carries the 60/25 kcal/mol rises, the 87x/100x ratio and the r_max line.
_ap = argparse.ArgumentParser()
_ap.add_argument("--annotate", action="store_true")
ANN = _ap.parse_args().annotate


def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")


S = pd.read_csv("MLIP_paper/falsify_label_uma.csv")
K = pd.read_csv("MLIP_paper/falsify_label_uma_curves.csv")
S["amp"] = np.maximum(S.rise_kcal.abs(), S.drop_kcal.abs())
get = lambda n: S[S.system == n].iloc[0]

# >>> SECOND ARCHITECTURE ADDED 2026-08-06. MACE-OMOL has charge and spin but NO
# composition embedding at all, so if it reproduces the pattern then composition
# is definitively not required -- the spin label alone does it. It does reproduce
# it: 87x on the spin-isolating triple against UMA's 100x.
try:
    O = pd.read_csv("MLIP_paper/falsify_label_omol.csv")
    O["amp"] = np.maximum(O.rise_kcal.abs(), O.drop_kcal.abs())
    getO = lambda n: O[O.system == n].iloc[0]
except FileNotFoundError:
    O, getO = None, None

TRUE_C, FALSE_C = "#0072B2", "#D55E00"
PRETTY = {"C2H6": "C₂H₆", "C2F4": "C₂F₄", "C2Cl4": "C₂Cl₄", "CH3NO2": "CH₃NO₂",
          "NH3-BH3": "NH₃–BH₃", "H2O-H2O": "H₂O–H₂O", "CH4-CH4": "CH₄–CH₄"}

# 2 x 2 grid: (a) and (b) side by side on top, (c) centred underneath, so each
# panel gets a readable width on a journal page instead of a third of a strip.
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12.0, 10.4))
gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.95],
                       hspace=0.30, wspace=0.9)
ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
      fig.add_subplot(gs[1, 1:3])]
LAB = dict(fontsize=13, fontweight="bold")
plt.rcParams["xtick.labelsize"] = 11; plt.rcParams["ytick.labelsize"] = 11

# ---------- (a) the four strong bonds -----------------------------------------
a = ax[0]
SHOW = [("C2H6",   FALSE_C,   "-", 2.8, "C₂H₆ → 2 CH₃•  (spin false, 176.6)"),
        ("C2Cl4",  TRUE_C,    "-", 2.5, "C₂Cl₄ → 2 CCl₂  (spin true, 119.5)"),
        ("CH3NO2", "#E8956A", "-", 2.3, "CH₃NO₂ → CH₃• + •NO₂  (spin false, 100.8)"),
        ("C2F4",   "#56B4E9", "-", 2.3, "C₂F₄ → 2 CF₂  (spin true, 78.0)")]
for n, c, ls, lw, lab in SHOW:
    d = K[K.system == n].sort_values("r_A")
    if d.empty: continue
    a.plot(d.r_A, d.E_rel_kcal, ls, color=c, lw=lw, label=lab, zorder=3)
a.axvline(6.0, color="#888", ls="--", lw=1.3)
a.axvspan(6.0, 8.05, color="#888888", alpha=.08)
if ANN:
    a.text(6.08, 8, "r_max = 6.0 Å", fontsize=9, color="#555")
    for n, xa, col in [("C2H6", 7.55, FALSE_C), ("CH3NO2", 7.15, "#E8956A")]:
        x = get(n)
        a.annotate("", xy=(xa, x.binding_kcal), xytext=(xa, x.binding_kcal - x.rise_kcal),
                   arrowprops=dict(arrowstyle="<->", color=col, lw=1.8))
        a.text(xa - .12, x.binding_kcal - x.rise_kcal / 2, f"{x.rise_kcal:.0f} ",
               ha="right", va="center", fontsize=9.5, fontweight="bold", color=col)
    a.text(7.5, 122, "flat", ha="center", fontsize=9, color=TRUE_C, fontweight="bold")
    a.set_title("(a)  Same envelope, same cutoff", loc="left", fontweight="bold")
a.set_xlim(1.0, 8.05); a.set_xlabel("separation (Å)", **LAB)
a.set_ylabel("energy rel. to own minimum (kcal/mol)", **LAB)
_panel_letter(a, "a")
# the curves top out near 180; the legend goes above them
a.set_ylim(top=300)
a.legend(fontsize=11.5, loc="upper right", framealpha=.95, handlelength=2.2,
         borderpad=0.6, labelspacing=0.45)
a.grid(alpha=.2)

# ---------- (b) ramp vs bond strength -----------------------------------------
b = ax[1]
for lt, c, mk, lab in [(True, TRUE_C, "o", "label stays TRUE"),
                       (False, FALSE_C, "D", "label becomes FALSE")]:
    g = S[S.label_true == lt]
    b.scatter(g.binding_kcal, g.amp, s=200, marker=mk, color=c,
              edgecolor="black", lw=.9, zorder=4, label=lab)
for _, x in S.iterrows():
    dy = 1.6 if x.system not in ("CH3NO2", "C2H6") else 0.66
    b.annotate(PRETTY[x.system], (x.binding_kcal, x.amp * dy), ha="center",
               fontsize=11, fontweight="bold" if x.binding_kcal > 70 else "normal")
b.set_yscale("log"); b.set_ylim(0.15, 200); b.set_xlim(-12, 200)
b.set_xlabel("bond strength (kcal/mol)", **LAB)
b.set_ylabel("spurious rise (kcal/mol)", **LAB)
_panel_letter(b, "b")
b.legend(fontsize=12, loc="upper left", framealpha=.95, borderpad=0.6,
         labelspacing=0.5)
b.grid(alpha=.2, which="both")
if ANN:
    b.set_title("(b)  The ramp does not track bond strength", loc="left", fontweight="bold")
    b.annotate("2nd strongest bond in the study,\nsmallest ramp",
               xy=(119.5, 0.60), xytext=(150, 0.26), fontsize=8.6, color=TRUE_C,
               fontweight="bold", ha="center",
               arrowprops=dict(arrowstyle="->", color=TRUE_C, lw=1.4))

# ---------- (c) the spin-isolating triple, BOTH architectures -----------------
c = ax[2]
TRIP = ["C2Cl4", "C2F4", "C2H6"]
vals = [get(n).amp for n in TRIP]
valsO = [getO(n).amp for n in TRIP] if O is not None else None
cols = [TRUE_C, "#56B4E9", FALSE_C]
if valsO is None:
    c.bar(range(3), vals, color=cols, edgecolor="black", lw=.7, width=.6)
    for i, v in enumerate(vals):
        c.text(i, v * 1.25, f"{v:.1f}", ha="center", fontweight="bold", fontsize=11)
else:
    w = .38
    x = np.arange(3)
    c.bar(x - w/2, valsO, w, color=cols, edgecolor="black", lw=.7, hatch="///",
          alpha=.8, label="MACE-OMOL-0")
    c.bar(x + w/2, vals, w, color=cols, edgecolor="black", lw=.7,
          label="UMA-S-1P1")
    for i in range(3):
        c.text(i - w/2, valsO[i] * 1.3, f"{valsO[i]:.1f}", ha="center",
               fontsize=11, fontweight="bold")
        c.text(i + w/2, vals[i] * 1.3, f"{vals[i]:.1f}", ha="center",
               fontsize=11, fontweight="bold")
    c.legend(fontsize=12, loc="upper left", framealpha=.95, borderpad=0.6,
             labelspacing=0.5)
c.set_yscale("log"); c.set_ylim(0.2, 400)
c.set_xticks(range(3))
c.set_xticklabels([f"{PRETTY[n]}\n→ 2 CCl₂\nspin TRUE" if n == "C2Cl4" else
                   f"{PRETTY[n]}\n→ 2 CF₂\nspin TRUE" if n == "C2F4" else
                   f"{PRETTY[n]}\n→ 2 CH₃•\nspin FALSE" for n in TRIP], fontsize=11.5)
c.set_ylabel("spurious rise (kcal/mol)", **LAB)
_panel_letter(c, "c")
c.grid(alpha=.2, axis="y", which="both")
_r = f"{valsO[2]/valsO[0]:.0f}× / {vals[2]/vals[0]:.0f}×" if valsO else f"{vals[2]/vals[0]:.0f}×"
if ANN:
    c.set_title("(c)  Symmetric homolyses only,\n      composition preserved in all three",
                loc="left", fontweight="bold", fontsize=11.5)
    c.annotate("", xy=(1.75, 30), xytext=(0.25, 30),
               arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
    c.text(1.0, 38, _r, ha="center", fontsize=12.5, fontweight="bold")
    c.text(1.15, 0.32, "only unpaired electrons differ", ha="center", fontsize=8.8,
           style="italic", color="#444")
    fig.suptitle("The falsification test: the ramp follows the LABEL, not the bond"
                 + (", replicated on two architectures" if O is not None else ""),
                 fontsize=15.5, fontweight="bold")
    fig.text(0.5, 0.013, "UMA-S-1P1, rigid scans. The carbenes' largest jumps still "
             "sit at 5.7–5.9 Å, right at r_max: the envelope closes for them too; "
             "there is simply nothing false for it to expose.",
             ha="center", fontsize=8.7, color="#666", style="italic")
    fig.tight_layout(rect=[0, .033, 1, .945])
else:
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.10)
print(f"ratio OMOL/UMA C2H6 vs C2Cl4: {_r}")
fig.savefig("MLIP_paper/fig_falsify.png", dpi=200)
fig.savefig("MLIP_paper/fig_falsify.pdf")
print(f"spin-isolating triple: C2Cl4 {vals[0]:.2f}  C2F4 {vals[1]:.2f}  "
      f"C2H6 {vals[2]:.2f}  ->  {vals[2]/vals[0]:.0f}x")
print("wrote MLIP_paper/fig_falsify.png")
