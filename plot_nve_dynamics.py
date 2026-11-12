#!/usr/bin/env python
"""
plot_nve_dynamics.py -- the spurious rise exerts force on a real trajectory
============================================================================

One microcanonical trajectory of a SINGLE ISOLATED nitromethane (7 atoms,
pbc="F F F"). The claim the figure has to support:

    the C-N bond repeatedly reached 6-7 A and was pulled back every time, and
    what pulled it back was the model, because nothing else was there.

>>> REWRITTEN 2026-09-24. THE PREVIOUS VERSION ARGUED FROM A DISTANCE.
    It plotted the minimum inter-fragment atom-atom distance (max 5.41 A) and
    compared it against "the rise begins at 5.70 A" -- a threshold read off the
    RIGID SCAN, which is parameterised by r(C-N), not by the closest contact.
    Two different coordinates, so the comparison was meaningless, and the
    conclusion drawn from it ("the trajectory never reached the rise") was
    wrong. See PROJECT_NOTES SS22, which retracts SS20.

    The fix is not to pick the other distance. It is to stop arguing from
    distances. This is an ISOLATED system, so the net force on the NO2 fragment
    IS the inter-fragment force, exactly:

        F_CH3 + F_NO2 = 0     (verified to 2.6e-6 eV/A on every frame)

    Project it on the CH3 -> NO2 centre-of-mass axis and the sign is the whole
    argument: negative means the model is pulling the fragments back together.

>>> WHY THERE IS NO ENERGY PANEL, AND IT IS NOT THE REASON YOU THINK
    The energy panel was dropped earlier because the "non-conservation" it was
    built to show turned out to be dt^2 truncation error. It stays dropped for
    a second, independent reason: potential energy binned by separation is FLAT
    to within noise (90.4 to 92.0 kcal/mol across 4.0-7.0 A). The
    inter-fragment interaction is a few tenths of a kcal/mol and the fragments'
    own vibrational energy swings by ~10, which buries it. An energy panel here
    would honestly look like no effect at all. The force does not have this
    problem because the internal forces cancel in the fragment sum.

>>> THE CUTOFF LINE IS THE MECHANISM, NOT DECORATION
    UMA's cutoff is 6.00 A, confirmed from backbone.cutoff (SS3.17). The
    attractive wall switches on between 6.2 and 6.7 A. The closest atom pair
    never exceeds 5.41 A, so this is not the "no edges remain" cliff -- it is
    the region where the C...N and C...O edges are cut one at a time.

>>> ONLY THE PRE-ISOMERISATION WINDOW IS ANALYSED
    After the nitro -> nitrite rearrangement, "the C-N bond" is not the
    dissociation coordinate any more and the CH3 + NO2 scan stops describing
    the system. Turning points are collected before that event only.

USAGE
    python plot_nve_dynamics.py --traj uma_nve_traj.xyz \\
        --out MLIP_paper/fig_nve_dynamics.png
"""

import argparse
import os
import sys

import numpy as np
KCAL_PER_EV = 23.0605      # 1 eV/A = 23.06 kcal mol^-1 A^-1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import poster_style
    poster_style.apply()
except ModuleNotFoundError:
    class _NoStyle:
        @staticmethod
        def figsize(w, h):
            return (w, h)
    poster_style = _NoStyle()
    print("\n" + "!" * 72)
    print("  poster_style.py NOT FOUND -- type will not match the other paper")
    print("  figures. Copy it next to this script and rerun.")
    print("!" * 72 + "\n")

C_SEP = "#1f4fd8"
C_CN = "#9fb3e8"
C_FORCE = "#c62828"
C_EVENT = "#2e8b30"
KCAL = 23.0605

# CH3 = C + the three H; NO2 = N + the two O. Fixed by the frame-0 molecule,
# which is intact nitromethane; the partition is only used before the
# rearrangement, so it does not need to track changing connectivity.
FRAG_A = [0, 4, 5, 6]
FRAG_B = [1, 2, 3]


def fragment_forces(atoms, F):
    """Net force on NO2, and the CH3 -> NO2 centre-of-mass unit vector.

    Returns (R_com, F_radial). F_radial < 0 means the fragments are being
    pulled back together. In an isolated system this is exactly the
    inter-fragment force -- the internal forces cancel in the sum.
    """
    m = atoms.get_masses()
    p = atoms.get_positions()
    cA = (p[FRAG_A] * m[FRAG_A, None]).sum(0) / m[FRAG_A].sum()
    cB = (p[FRAG_B] * m[FRAG_B, None]).sum(0) / m[FRAG_B].sum()
    d = cB - cA
    R = np.linalg.norm(d)
    return R, float(np.dot(F[FRAG_B].sum(0), d / R))



def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style.
    Outside the axes so it can never land on data, a legend or a colourbar."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="uma_nve_traj.xyz")
    ap.add_argument("--frame-ps", type=float, default=0.01,
                    help="logged frame spacing; nve_log.csv says 0.01 ps")
    ap.add_argument("--cutoff", type=float, default=6.00,
                    help="model cutoff in A; UMA is 6.00 (backbone.cutoff)")
    ap.add_argument("--exc-min", type=float, default=5.5,
                    help="COM separation an excursion must exceed to count")
    ap.add_argument("--fmin", type=float, default=3.5,
                    help="panel (b) starts here. Inside ~3 A the C-N bond is "
                         "still formed and its restoring force is tens of "
                         "times larger than the effect being shown, so "
                         "including it sets the y-scale and hides everything.")
    ap.add_argument("--coord", choices=["rcn", "com"], default="rcn",
                    help="x coordinate for both panels. 'rcn' is r(C-N), the "
                         "rigid scan's coordinate (Figure 1), so the force "
                         "onset here and the scan's rise share one axis. In "
                         "the rigid scan R_com = r(C-N) + 0.46 A exactly; in "
                         "the NVE the fragments tumble and the offset averages "
                         "0.06 A, so the two coordinates are NOT "
                         "interchangeable between the two experiments.")
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: titles, cutoff labels, on-axes notes")
    ap.add_argument("--out", default="MLIP_paper/fig_nve_dynamics.png")
    a = ap.parse_args()

    from ase.io import read
    tr = read(a.traj, ":")
    if tr[0].pbc.any():
        sys.exit(f"  {a.traj} is periodic. This analysis assumes an isolated\n"
                 f"  molecule: with images or neighbours present, the net force\n"
                 f"  on NO2 is NOT the inter-fragment force and the whole\n"
                 f"  argument collapses.")

    t = np.arange(len(tr)) * a.frame_ps
    pos = np.array([at.get_positions() for at in tr])
    F = np.array([at.get_forces() for at in tr])
    r_cn = np.array([at.get_distance(0, 1) for at in tr])

    # >>> NEWTON'S THIRD LAW IS THE LOAD-BEARING ASSUMPTION. CHECK IT.
    resid = np.abs(F[:, FRAG_A, :].sum(1) + F[:, FRAG_B, :].sum(1)).max()
    print(f"  |F_CH3 + F_NO2| max = {resid:.2e} eV/A", end="")
    if resid > 1e-3:
        sys.exit("\n  net force is not zero: the system is not isolated, or the\n"
                 "  forces in this file are not the ones that generated it.")
    print("   (isolated, as required)")

    R = np.empty(len(tr))
    Frad = np.empty(len(tr))
    # >>> HOW MANY OF THE 12 INTER-FRAGMENT ATOM PAIRS ARE INSIDE THE CUTOFF.
    # This is the variable the model actually responds to. The scan's rise is
    # these twelve pairs leaving the 6.0 A neighbour list one at a time
    # (H...O first near 4.7 A, C...N last at 6.0, in the scan's geometry), and
    # the scan is flat beyond 6.0 A because by then ALL twelve have left. In
    # the trajectory the fragments tumble, so r(C-N) = 6.8 A can still have a
    # hydrogen within 5 A of an oxygen. Colouring by this count shows that the
    # trajectory never reached the scan's flat region at all.
    npairs = np.empty(len(tr), int)
    for k, at in enumerate(tr):
        R[k], Frad[k] = fragment_forces(at, F[k])
        Frad[k] *= KCAL_PER_EV          # paper units: kcal mol^-1 A^-1
        D = np.linalg.norm(pos[k][FRAG_A][:, None, :] - pos[k][FRAG_B][None, :, :],
                           axis=-1)
        npairs[k] = int((D < a.cutoff).sum())

    # ---- events, from geometry ------------------------------------------
    # nitro -> nitrite: a C-O contact forms and holds
    co = np.min(np.linalg.norm(pos[:, [0], :] - pos[:, [2, 3], :], axis=-1), axis=1)
    hit = np.where(co < 1.6)[0]
    t_iso = t[hit[0]] if len(hit) else np.inf
    # H transfer: an H closer to N or O than to C, sustained
    dh_c = np.linalg.norm(pos[:, [4, 5, 6], :] - pos[:, [0], :], axis=-1)
    dh_x = np.min(np.stack([np.linalg.norm(pos[:, [4, 5, 6], :] - pos[:, [j], :],
                                           axis=-1) for j in (1, 2, 3)]), axis=0)
    moved = (dh_x < dh_c).any(axis=1)
    run = np.convolve(moved.astype(int), np.ones(20), "same")
    ht = np.where(run >= 20)[0]
    t_h = t[ht[0]] if len(ht) else np.inf
    print(f"  nitro->nitrite at {t_iso:.2f} ps;  H transfer at {t_h:.2f} ps")

    pre = t < min(t_iso, t_h)

    # >>> ONE COORDINATE FOR BOTH PANELS, AND IT IS THE SCAN'S.
    # The force itself is still the net force on NO2 projected on the
    # centre-of-mass axis (exact for an isolated system). Only the x axis
    # changes: r(C-N) puts this figure on the same axis as Figure 1, so the
    # scan's rise (5.3-6.0 A) and this force onset can be read against each
    # other directly. See --coord.
    X = r_cn if a.coord == "rcn" else R
    xname = "r(C–N) (Å)" if a.coord == "rcn" else "CH₃···NO₂ centre-of-mass separation (Å)"
    print(f"  coordinate: {xname};  mean(R_com - r_CN) over the window = "
          f"{(R - r_cn)[pre].mean():.3f} A (rigid scan: 0.459)")

    # ---- turning points --------------------------------------------------
    idx = np.where(pre & (X > a.exc_min))[0]
    groups = ([g for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
               if len(g)] if len(idx) else [])
    tps = [g[int(np.argmax(X[g]))] for g in groups]
    print(f"\n  {len(tps)} excursions past {a.coord} = {a.exc_min} A before the "
          f"rearrangement")
    print(f"  {'t (ps)':>8}{'R_com':>8}{'r(C-N)':>9}{'F_radial':>11}")
    for k in tps:
        print(f"  {t[k]:>8.2f}{R[k]:>8.2f}{r_cn[k]:>9.2f}{Frad[k]:>11.3f}")
    n_att = sum(1 for k in tps if Frad[k] < 0)
    print(f"  attractive at {n_att} of {len(tps)} turning points")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=poster_style.figsize(13.8, 5.6))
    # right margin holds the colourbar AND its label; 0.99 clipped both
    fig.subplots_adjust(top=0.86 if a.annotate else 0.92, bottom=0.135,
                        left=0.07, right=0.92, wspace=0.22)

    # ---- (a) separation against time -------------------------------------
    # the shaded band and the dashed line stay: they are reference marks the
    # caption explains. The red label, the titles and the notes are --annotate.
    ax1.axhspan(a.cutoff, max(X[pre].max(), 7.2) * 1.02, color="#c62828",
                alpha=.10, zorder=0)
    ax1.axhline(a.cutoff, color=C_FORCE, ls="--", lw=1.6, zorder=2)
    if a.annotate:
        ax1.text(t[pre][-1] * 0.015, a.cutoff + 0.12,
                 f"model cutoff {a.cutoff:.2f} Å", fontsize=9.5, color=C_FORCE,
                 ha="left", va="bottom", fontweight="bold")
    ax1.plot(t[pre], X[pre], color=C_SEP, lw=2.4, zorder=4, label=xname)
    ax1.plot(t[tps], X[tps], "v", color=C_FORCE, ms=11, markeredgecolor="white",
             markeredgewidth=1.3, zorder=6,
             label=f"turning point ({n_att}/{len(tps)} attractive)")
    ax1.set_xlim(0, t[pre][-1])
    ax1.set_ylim(0, max(X[pre].max(), 7.2) * 1.02)
    ax1.set_xlabel("time (ps)", fontsize=11)
    ax1.set_ylabel(xname, fontsize=11)
    if a.annotate:
        ax1.set_title("(a)  it reaches 6–7 Å and comes back, every time",
                      loc="left", fontweight="bold", fontsize=11.5)
    ax1.grid(alpha=.2)
    ax1.legend(fontsize=14, loc="lower right", framealpha=.95)
    _panel_letter(ax1, "a")

    # ---- (b) the force that brings it back -------------------------------
    # >>> THIS PANEL IS THE ARGUMENT. Everything else is context.
    far = pre & (X >= a.fmin)
    ax2.axhline(0, color="0.4", lw=1.2, zorder=2)
    ax2.axvline(a.cutoff, color=C_FORCE, ls="--", lw=1.6, zorder=2)
    sc = ax2.scatter(X[far], Frad[far], s=16, c=npairs[far], cmap="viridis",
                     vmin=0, vmax=12, zorder=3, linewidths=0,
                     label="every frame")
    ax2.scatter(X[tps], Frad[tps], s=110, c=npairs[tps], cmap="viridis",
                vmin=0, vmax=12, edgecolor=C_FORCE, lw=1.8, zorder=6,
                label="turning points")
    cb = fig.colorbar(sc, ax=ax2, pad=0.02, fraction=0.05)
    cb.set_label("atom pairs within cutoff (of 12)", fontsize=10)
    cb.set_ticks([0, 3, 6, 9, 12])
    n_min = int(npairs[pre].min())
    print(f"  fewest pairs inside the cutoff in any pre-isomerisation frame: "
          f"{n_min} of 12  (the scan's flat region is 0; reached in "
          f"{int((npairs[pre] == 0).sum())} frames)")

    # binned mean, so the trend is readable through the scatter
    # 0.3 A bins: fine enough to resolve the 5.6-5.9 A onset against the scan
    edges = np.arange(a.fmin, X[pre].max() + 0.3, 0.3)
    cen, mean = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = pre & (X >= lo) & (X < hi)
        if m.sum() >= 5:
            cen.append((lo + hi) / 2)
            mean.append(Frad[m].mean())
    ax2.plot(cen, mean, color="0.15", lw=2.6, zorder=5, marker="o", ms=5,
             label="mean in 0.3 Å bins")
    ax2.set_xlim(a.fmin, max(X[pre].max(), 7.0) * 1.02)
    # y-range from the plotted window only, so the bonded region's much larger
    # restoring force cannot set the scale
    lo_y = min(Frad[far].min(), min(mean)) * 1.15
    ax2.set_ylim(lo_y, max(Frad[far].max(), 0.4 * KCAL_PER_EV) * 1.5)
    if a.annotate:
        ax2.text(a.cutoff - 0.04, lo_y * 0.97, f"cutoff {a.cutoff:.2f} Å",
                 fontsize=9.5, color=C_FORCE, rotation=90, ha="right",
                 va="bottom", fontweight="bold")
    ax2.set_xlabel(xname, fontsize=11)
    ax2.set_ylabel("inter-fragment force (kcal mol$^{-1}$ Å$^{-1}$)", fontsize=11)
    if a.annotate:
        ax2.set_title("(b)  attraction that grows with distance", loc="left",
                      fontweight="bold", fontsize=11.5)
    ax2.grid(alpha=.2)
    ax2.legend(fontsize=14, loc="lower left", framealpha=.95)
    _panel_letter(ax2, "b")
    if a.annotate:
        ax2.annotate("negative = pulled back together", xy=(0.97, 0.06),
                     xycoords="axes fraction", fontsize=9.5, color="0.35",
                     ha="right", style="italic")
        fig.suptitle("An isolated molecule cannot escape its own potential",
                     fontsize=14.5, fontweight="bold", y=0.975)
    # onset: first bin (outward) whose mean force is below -2.3 kcal/mol/A (-0.1 eV/A)
    on = next((c for c, m in zip(cen, mean) if m < -0.1 * KCAL_PER_EV), None)
    print(f"  force onset (first 0.3 A bin with mean < -2.3 kcal/mol/A): "
          f"{on - 0.15:.1f}-{on + 0.15:.1f} A in {a.coord}" if on else
          "  no bin below -2.3 kcal/mol/A")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"\n  wrote {a.out}")

    print("\n  CAPTION MUST STATE:")
    print(f"    system         one nitromethane, 7 atoms, no periodic images, "
          f"no solvent")
    print(f"    window         0 to {t[pre][-1]:.2f} ps, before the nitro→nitrite "
          f"rearrangement at {t_iso:.2f} ps")
    print( "    force          net force on NO₂ projected on the CH₃→NO₂ COM "
           "axis; in an")
    print( "                   isolated system this is exactly the "
           "inter-fragment force")
    print(f"    scale          two neutral radicals at 6.5 Å interact at "
          f"~0.02 kcal/mol/Å;")
    print(f"                   the binned mean here reaches "
          f"{min(mean):.1f} kcal/mol/Å")
    print(f"    cutoff         {a.cutoff:.2f} Å, from backbone.cutoff, not fitted "
          f"to this plot")
    print( "    NOT SHOWN      potential energy, which is flat to within noise "
           "across this")
    print( "                   range: fragment internal vibration swamps the "
           "interaction")


if __name__ == "__main__":
    main()
