#!/usr/bin/env python
"""
plot_reactant_survival.py -- every model on one axis: how fast is CH3NO2 consumed?
==================================================================================

The decomposition-product figure for the Results section. One panel, one curve
per model, nitromethane population against time.

>>> WHY ONE PANEL AND NOT FIVE
    The per-model Han Fig. 5 plots carry five products each, and the products all
    end up in much the same place -- the note at the top of plot_decomp_compare.py
    says so. The only curve that separates the models is the REACTANT. Two of the
    five runs also die before any chemistry starts (MACE-OFF23 still holds
    51-64% of its nitromethane at 2.4 ps), so their panels would be nearly empty.
    Overlaying the reactant puts survival and rate in the same view, and the
    product slate goes in a table where exact numbers belong.

>>> WHY LOG TIME, AND WHEN NOT TO USE IT
    The runs span 2.4 to 150 ps, a factor of sixty. On a linear axis MACE-OFF23's
    entire lifetime is the first 1.6% of the plot and ANI-1xnr's sub-picosecond
    consumption is invisible. On a log axis they share a view with the 150 ps
    runs. A curve simply STOPS where its run died, so the reader sees stability
    and rate at once without a second figure.

    The cost is real and worth stating. A log axis makes a first-order decay,
    which is a straight line on a semi-log plot, into a sigmoid, so the shape a
    kinetics reader looks for is not the shape they see. It also stretches the
    equilibration window into a third of the canvas, which is the least
    interesting third. --linear-time gives the conventional axis; use it for a
    figure about RATES, and log for a figure about the sixty-fold spread in
    survival. The honest arrangement, if there is room, is both: log for the
    overview and a linear inset or companion panel over 0-20 ps for the decay.

>>> WHY EQUILIBRATION IS INCLUDED BY DEFAULT
    ANI-1xnr arrives at production already ~91% decomposed: it destroys the box
    during a 300 K equilibration, before the temperature is raised at all. A plot
    that starts at production t=0 shows that as a flat line near 0.09 with no
    explanation, and hides the single most dramatic result in the comparison. So
    time runs from the START OF EQUILIBRATION and the equilibration window is
    shaded.

    CAVEAT THE SHADING HIDES, SO STATE IT IN THE CAPTION: equilibration is 2 ps
    at 300 K for the liquid and 5 ps at 50 K for the crystal. Those are different
    conditions from the 3000 K production that follows, and from each other. The
    shaded band is drawn at the LONGEST equilibration among the runs plotted, so
    for liquid-only figures it is exact and for mixed figures it is an upper
    bound. --no-eq drops it and plots production alone.

>>> HALF-LIFE IS MEASURED AGAINST 1.0, NOT AGAINST y[0]
    Every box starts with every molecule intact, so y0 = 1.0 is a known initial
    condition. Measuring "half of y[0]" instead returns 0.00 ps for a fast decay,
    because a forward-averaging rolling window has already dropped the first
    point below 0.5. That bug produced a published-looking 0.00 ps half-life for
    ANI-1xnr once. The rolling window here is deliberately short for the same
    reason: 0.25 ps, not the 2 ps used for product plateaus.

USAGE
    python plot_reactant_survival.py --runs ani_run omol_run uma_run \\
        off23_run_cleanbox --out MLIP_paper/reactant_survival.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# >>> poster_style IS OPTIONAL, BUT ITS ABSENCE IS NOT SILENT.
# This script gets copied to the cluster on its own and poster_style.py stays
# behind, which is a hard ImportError for a module that only sets font sizes.
# Falling back keeps the analysis runnable. Doing so QUIETLY would be worse
# than failing, because the figure would come out in default matplotlib type
# and not match any other figure in the paper -- and nobody notices that until
# the figures are side by side in a draft.
try:
    import poster_style
    poster_style.apply()
    _HAVE_STYLE = True
except ModuleNotFoundError:
    _HAVE_STYLE = False

    class _NoStyle:
        @staticmethod
        def figsize(w, h):
            return (w, h)
    poster_style = _NoStyle()
    print("\n" + "!" * 72)
    print("  poster_style.py NOT FOUND -- falling back to matplotlib defaults.")
    print("  The figure will RUN but its type will not match the other paper")
    print("  figures. Copy poster_style.py next to this script and rerun before")
    print("  using the output in the manuscript.")
    print("!" * 72 + "\n")

ATOMS_PER_NM = 7
REACTANT = "CH3NO2 (NM+isomers)"

# >>> 0.25 ps, NOT 2 ps. See the half-life note above. A 2 ps forward window
# destroys any sub-picosecond decay before it can be measured.
REACT_ROLL = 0.25

# Colour and label by model family, so the eye groups OMol-derived models
# together and the reactive model stands apart.
STYLE = {
    "ani":    ("#b03030", "ANI-1xnr"),
    "ani_mrr": ("#0f7d7d", "ANI-1xnr-Mrr"),
    "off23":  ("#e8860c", "MACE-OFF23"),
    # the crystal run is named off_run_crystal_clean, without the "23"; classify
    # tries the longest key first, so "off23" still wins where it applies
    "off":    ("#e8860c", "MACE-OFF23"),
    "osaka":  ("#2e8b30", "MACE-Osaka24"),
    "omol":   ("#1f4fd8", "MACE-OMOL-0"),
    "uma":    ("#8e6bbf", "UMA-S-1P1"),
}


def classify(name):
    """Map a run directory name onto a model style. Longest key first, so
    'ani_mrr_run' does not match the 'ani' rule."""
    n = name.lower()
    for key in sorted(STYLE, key=len, reverse=True):
        if n.startswith(key):
            return STYLE[key]
    return ("0.4", name)


def find(d, patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    return None


def half_life(t, y, y0=1.0):
    w = np.where(np.asarray(y) <= 0.5 * y0)[0]
    return float(t[w[0]]) if len(w) else np.nan


# >>> TIMING IS A PROPERTY OF THE RUN, NOT OF THE COMMAND LINE.
# --dt-fs 0.25 --log-stride 100 was applied to every run at once. osaka_run's
# liquid production used dt = 0.5 fs, so its entire time axis was drawn at HALF
# its true length: a 10.0 ps run that completed normally appeared on the figure
# as a run that "died at 5.1 ps". Nothing errored and nothing looked wrong.
#
# Every run writes a log whose second column IS the time in ps, so the spacing
# can be read rather than assumed. The frame count is the cross-check:
# run_reactive_md.py writes frame 0 before stepping, so a trajectory should
# hold exactly one more frame than its log holds data lines. If it does not,
# the trajectory and the log were written at different strides and the log's
# spacing does not describe the trajectory.
MD_LOGS = {"production": ("production.log", "production_*.log", "*prod*.log"),
           "equilibration": ("equilibration.log", "*equil*.log", "*eq*.log")}


def frame_spacing_ps(d, kind, n_frames):
    """Picoseconds per frame for one phase of a run, read from its log.

    Returns (frame_ps, note). frame_ps is None when the log cannot settle the
    question, and the caller must then fall back to the command line AND SAY SO.
    """
    log = find(d, MD_LOGS[kind])
    if log is None:
        return None, f"no {kind} log"
    times = []
    for line in open(log):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 2:
            continue
        try:
            times.append(float(f[1]))
        except ValueError:
            continue
        if len(times) >= 2:
            break
    if len(times) < 2:
        return None, f"{os.path.basename(log)} has fewer than 2 data lines"
    dt = times[1] - times[0]
    if dt <= 0:
        return None, f"{os.path.basename(log)} time column is not increasing"

    n_lines = sum(1 for ln in open(log)
                  if ln.strip() and not ln.startswith("#"))
    note = f"{os.path.basename(log)}: {dt*1000:.0f} fs/frame"
    if n_frames is not None and n_frames != n_lines + 1:
        note += (f"  [WARNING: {n_frames} frames but {n_lines} log lines; "
                 f"trajectory and log strides may differ]")
    return dt, note


M_NITROMETHANE = 61.04       # g/mol
N_AVOGADRO = 6.02214076e23


try:
    from run_manifest import read_manifest, fingerprint as box_fingerprint
except ImportError:                       # older checkout, cluster copy, etc.
    def read_manifest(_):
        return None

    def box_fingerprint(atoms):
        from ase.data import covalent_radii
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        rc = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
        A = d < 1.15 * (rc[:, None] + rc[None, :])
        n = len(atoms)
        seen = np.zeros(n, bool)
        comps = []
        for i in range(n):
            if seen[i]:
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                k = stack.pop()
                comp.append(k)
                for j in np.where(A[k] & ~seen)[0]:
                    seen[j] = True
                    stack.append(j)
            comps.append(comp)
        inter = d.copy()
        for c in comps:
            inter[np.ix_(c, c)] = np.inf
        return {"n_fragments": len(comps),
                "min_intermolecular_A": float(inter.min())}


def density_g_cc(atoms):
    """Mass density of a box of whole nitromethane molecules, from frame 0.

    The starting density is the single number that says whether two runs are
    the same experiment. It is not recorded in the trajectory metadata, but it
    is implied exactly by the cell and the atom count.
    """
    v_cm3 = atoms.get_volume() * 1e-24
    return (len(atoms) // ATOMS_PER_NM) * M_NITROMETHANE / N_AVOGADRO / v_cm3


def load_run(d, a):
    from ase.io.trajectory import Trajectory
    from species_analysis import species_populations

    prod = find(d, ("production.traj", "production_*.traj", "*prod*.traj"))
    if prod is None:
        return None

    # >>> IS THIS RUN ACTUALLY FINISHED?
    # run_reactive_md.py writes final.xyz only at the end. A run still in
    # progress has a valid, growing production.traj, and this figure would draw
    # it as a curve that stops at a dot -- which is the visual language this
    # figure uses for A RUN THAT DIED. That is not a cosmetic problem: it would
    # put a false stability failure in the paper. The ani_run row in the first
    # product_slate table read 9.7 ps for exactly this reason, because the file
    # was read while the job was still writing to it.
    finished = os.path.exists(os.path.join(d, "final.xyz"))
    name = os.path.basename(d.rstrip("/"))
    warn = []

    prod_frames = list(Trajectory(prod))
    dt_prod, note = frame_spacing_ps(d, "production", len(prod_frames))
    if dt_prod is None:
        dt_prod = a.dt_fs * a.log_stride / 1000.0
        warn.append(f"production timing ASSUMED from the command line "
                    f"({dt_prod*1000:.0f} fs/frame): {note}")
    elif "WARNING" in note:
        warn.append(note)

    rho = density_g_cc(prod_frames[0])

    # >>> IS THE equilibration.traj IN THIS FOLDER THIS RUN'S EQUILIBRATION?
    # It is not always. osaka_run holds a 4-frame equilibration.traj of the
    # 1.97 g/cc box that blew up at 0.05 ps, next to a production trajectory of
    # the 1.137 g/cc box that ran cleanly for 10 ps. They are two different
    # jobs that happen to share a directory. Concatenated, the abandoned
    # equilibration's explosion is drawn as the production run's first
    # picosecond of chemistry -- a decay to zero that never happened.
    #
    # The cell is the discriminator and it is exact: a run cannot change its
    # own cell under NVT, so an equilibration whose cell differs from
    # production frame 0 is not that production run's equilibration.
    eq_frames = []
    eq = find(d, ("equilibration.traj", "*eq*.traj")) if not a.no_eq else None
    if eq:
        cand = list(Trajectory(eq))
        c_eq = np.asarray(cand[0].cell.lengths())
        c_pr = np.asarray(prod_frames[0].cell.lengths())
        dt_e, note_e = frame_spacing_ps(d, "equilibration", len(cand))
        if not np.allclose(c_eq, c_pr, atol=1e-3):
            warn.append(
                f"DROPPED {os.path.basename(eq)}: its cell is "
                f"{c_eq[0]:.3f} A ({density_g_cc(cand[0]):.3f} g/cc) but "
                f"production frame 0 is {c_pr[0]:.3f} A ({rho:.3f} g/cc). "
                f"Different box, so it is not this run's equilibration.")
        elif dt_e is not None and abs(dt_e - dt_prod) > 1e-9:
            warn.append(
                f"DROPPED {os.path.basename(eq)}: {dt_e*1000:.0f} fs/frame "
                f"against production's {dt_prod*1000:.0f} fs/frame. One time "
                f"axis cannot carry two frame spacings.")
        else:
            eq_frames = cand

    # >>> WHICH BOX, NOT JUST WHICH DENSITY.
    # nm_box_1p97.xyz and nm_box_1p97_clean.xyz are BOTH 11.808 A and BOTH
    # 1.970 g/cc. The density guard below cannot tell them apart, and they are
    # not the same experiment: the first fuses two nitromethanes with an O-O
    # contact at 1.469 A and has a closest intermolecular contact of 1.290 A,
    # against 1.759 A for the clean one.
    #
    # The starting box is frame 0 of EQUILIBRATION, never of production, which
    # has already had 2 ps of dynamics applied to it. Order of preference:
    #   1. run_manifest.json  -- written by the run, carries the box sha256
    #   2. equilibration frame 0 fingerprint
    #   3. nothing, and say so
    start_box = None
    man = read_manifest(d)
    if man and man.get("fingerprint"):
        f = man["fingerprint"]
        start_box = dict(
            key=(f["n_fragments"], round(f["min_intermolecular_A"], 3)),
            source="run_manifest.json",
            name=os.path.basename(man.get("box", {}).get("path") or "?"),
            sha=(man.get("box", {}).get("sha256") or "")[:12],
            min_inter=f["min_intermolecular_A"], n_frag=f["n_fragments"])
    elif eq_frames:
        f = box_fingerprint(eq_frames[0])
        start_box = dict(
            key=(f["n_fragments"], round(f["min_intermolecular_A"], 3)),
            source="equilibration frame 0", name="(not recorded)", sha="",
            min_inter=f["min_intermolecular_A"], n_frag=f["n_fragments"])
    else:
        warn.append("starting box UNKNOWN: no run_manifest.json and no usable "
                    "equilibration trajectory. Production frame 0 has already "
                    "been equilibrated and cannot identify the box.")

    traj = (eq_frames[::a.stride] if eq_frames else []) + prod_frames[::a.stride]
    n_eq = len(eq_frames[::a.stride]) if eq_frames else 0

    good = [i for i, at in enumerate(traj)
            if np.isfinite(at.get_positions()).all()]
    traj = [traj[i] for i in good]

    n_nm = len(traj[0]) // ATOMS_PER_NM
    frame_ps = dt_prod * a.stride

    # species_populations builds its own time axis from dt_fs * log_stride, so
    # handing it the measured spacing directly (log_stride=1) is what keeps the
    # rolling window and the x-axis on the run's real clock.
    t, pops = species_populations(traj, dt_fs=frame_ps * 1000.0, log_stride=1,
                                  roll_ps=REACT_ROLL, scale=a.scale,
                                  n_nitromethane=n_nm)
    t = np.asarray(t, float)
    y = np.asarray(pops[REACTANT], float)

    # >>> THE ORIGIN IS CORRECT ON THE STRIDED AXIS. DO NOT "FIX" IT.
    # species_populations builds t = arange(n) * frame_ps over the concatenated
    # [eq[::stride] + prod[::stride]], so production frame j sits at index
    # n_eq + j and t_prod = t - n_eq*frame_ps = j*frame_ps, which is exactly
    # the real time since production start whatever the stride. Replacing this
    # with len(eq_frames)*dt_prod looks more principled and is WRONG: it
    # measures the equilibration in unstrided time while the axis is strided,
    # and shifts every half-life by the difference.
    #
    # What IS off by up to one strided frame is the WIDTH OF THE SHADED BAND:
    # 81 eq frames at 0.025 ps is 2.025 ps, drawn as 17 frames at 0.125 ps =
    # 2.125 ps. Cosmetic, and eq_ps_true below is what the caption should quote.
    eq_ps = n_eq * frame_ps
    eq_ps_true = len(eq_frames) * dt_prod if eq_frames else 0.0

    # >>> THE HALF-LIFE MUST NOT DEPEND ON --stride, AND IT DID.
    # The plotted series is strided, so the first crossing of 0.5 can only be
    # reported on a grid of frame_ps = dt_prod * stride. At --stride 5 that is
    # 0.125 ps, and a true crossing at 0.26 ps gets reported as 0.375 -- the
    # "0.38 ps" that went into the draft for ani_run, against a true 0.25.
    # Striding coarser can only ever push it later, never earlier, so every
    # fast decay was biased long.
    #
    # Fix: measure t(1/2) on an UNSTRIDED early window of production, whatever
    # the plotting stride. Only the first few ps are needed, so this stays
    # cheap even when the full run is 150 ps. Production frame 0 is t = 0 here,
    # so no origin arithmetic is involved either.
    n_early = min(len(prod_frames), int(a.half_life_ps / dt_prod) + 1)
    te, pe = species_populations(prod_frames[:n_early],
                                 dt_fs=dt_prod * 1000.0, log_stride=1,
                                 roll_ps=REACT_ROLL, scale=a.scale,
                                 n_nitromethane=n_nm)
    t_half = half_life(np.asarray(te, float),
                       np.asarray(pe[REACTANT], float))
    # time since production start, for the half-life; the plot uses time since
    # the start of equilibration so a log axis is possible
    t_prod = t - eq_ps
    return dict(name=name, t=t, y=y, n_mol=n_nm, rho=rho, frame_ps=frame_ps,
                eq_ps=eq_ps, eq_ps_true=eq_ps_true,
                t_end=float(t[-1]), finished=finished,
                warn=warn, start_box=start_box, t_half=t_half,
                half_life_res_ps=dt_prod,
                y_at_prod=float(y[np.argmin(np.abs(t_prod))]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dt-fs", type=float, default=0.25,
                    help="fallback only; timing is read from each run's log")
    ap.add_argument("--log-stride", type=int, default=100,
                    help="fallback only; see --dt-fs")
    ap.add_argument("--stride", type=int, default=5,
                    help="thins the PLOTTED curve only; the half-life is "
                         "measured unstrided, see --half-life-ps")
    ap.add_argument("--half-life-ps", type=float, default=20.0,
                    help="how much production to scan at FULL time resolution "
                         "for the half-life. Every run here crosses well "
                         "inside 20 ps; raise it for a slower model.")
    ap.add_argument("--scale", type=float, default=1.2,
                    help="bond threshold; the reactant is far less sensitive "
                         "to this than H2O is, but state it in the caption")
    ap.add_argument("--no-eq", action="store_true",
                    help="plot production only; see the equilibration note")
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--mixed-density", action="store_true",
                    help="allow runs at different starting densities on one "
                         "axis; only with a caption that says so")
    ap.add_argument("--mixed-box", action="store_true",
                    help="allow runs from different starting structures at the "
                         "same density; only with a caption that says so")
    ap.add_argument("--tmax", type=float, default=None,
                    help="crop the x-axis to this many ps from the start of "
                         "equilibration. A two-run comparison whose decays "
                         "both finish inside 3 ps spends 98%% of a 150 ps "
                         "canvas on a flat line at zero; crop it.")
    ap.add_argument("--linear-time", action="store_true",
                    help="linear x-axis; see the log-time note above")
    ap.add_argument("--title", default="",
                    help="on-figure title; empty for publication")
    ap.add_argument("--out", default="MLIP_paper/reactant_survival.png")
    a = ap.parse_args()

    runs = []
    for d in sorted(x for x in a.runs if os.path.isdir(x)):
        print(f"  [{os.path.basename(d.rstrip('/')):<22}] ", end="", flush=True)
        r = load_run(d, a)
        if r is None:
            print("no production trajectory")
            continue
        print(f"{r['n_mol']} mol, {r['rho']:.3f} g/cc, "
              f"{r['frame_ps']*1000/a.stride:.0f} fs/frame x{a.stride}, eq "
              f"{r['eq_ps']:.1f} ps, ends {r['t_end']:.1f} ps,"
              f"  t(1/2) = {r['t_half']:+.2f} ps,"
              f"  intact at production start = {r['y_at_prod']:.2f}"
              + ("" if r["finished"] else "   <-- NO final.xyz: STILL RUNNING?"))
        for w in r["warn"]:
            print(f"      !! {w}")
        runs.append(r)
    if not runs:
        sys.exit("  nothing to plot")

    # >>> RUNS AT DIFFERENT DENSITIES ARE NOT A COMPARISON.
    # Nitromethane's decomposition rate depends strongly on density, so an
    # ambient-density run (1.14 g/cc) and a compressed run (1.97 g/cc) on the
    # same axis compare two experiments, not two models, and the figure's own
    # caption would state a density that is wrong for one of the curves. This
    # stops rather than warns, because the resulting figure looks entirely
    # reasonable and there is nothing on it to give the mistake away.
    rhos = sorted({round(r["rho"], 3) for r in runs})
    if len(rhos) > 1 and not a.mixed_density:
        print("\n" + "!" * 72)
        print("  RUNS AT DIFFERENT DENSITIES:")
        for r in sorted(runs, key=lambda r: -r["rho"]):
            print(f"    {r['name']:<26} {r['rho']:.3f} g/cc   "
                  f"{r['n_mol']} molecules")
        print("  These are different experiments. Drop the odd run, or pass")
        print("  --mixed-density if the difference is deliberate and the")
        print("  caption says so.")
        print("!" * 72)
        sys.exit(1)
    rho_label = (f"{rhos[0]:.2f} g/cc" if len(rhos) == 1
                 else "mixed density (see caption)")

    # >>> SAME DENSITY IS NOT SAME BOX.
    # This guard exists because the density check passed on a figure that mixed
    # nm_box_1p97.xyz with nm_box_1p97_clean.xyz -- identical cell, identical
    # density, different structures. The closest intermolecular contact
    # separates them by half an angstrom, which is far outside anything frame 0
    # could vary by.
    known = [r for r in runs if r.get("start_box")]
    unknown = [r for r in runs if not r.get("start_box")]
    boxes = {}
    for r in known:
        boxes.setdefault(r["start_box"]["key"], []).append(r)

    if unknown:
        print("\n  !! starting box could not be identified for: "
              + ", ".join(r["name"] for r in unknown))
        print("     Rerun those with run_manifest.py in place, or keep their "
              "equilibration.traj.")

    if len(boxes) > 1 and not a.mixed_box:
        print("\n" + "!" * 72)
        print("  RUNS STARTED FROM DIFFERENT BOXES (same density, so the")
        print("  density check above passed):")
        for key, rs in sorted(boxes.items(), key=lambda kv: -kv[0][1]):
            b = rs[0]["start_box"]
            print(f"    {b['n_frag']} fragments, closest intermolecular "
                  f"{b['min_inter']:.3f} A   [{b['source']}"
                  + (f", {b['name']} {b['sha']}]" if b["sha"] else "]"))
            for r in rs:
                print(f"       {r['name']}")
        print("  A model comparison requires one starting structure. Rerun the")
        print("  odd ones, or pass --mixed-box with a caption that says so.")
        print("!" * 72)
        sys.exit(1)

    if len(boxes) == 1:
        b = next(iter(boxes.values()))[0]["start_box"]
        print(f"\n  starting box: all {len(known)} run(s) agree — "
              f"{b['n_frag']} fragments, closest intermolecular "
              f"{b['min_inter']:.3f} A"
              + (f", {b['name']}" if b["name"] != "(not recorded)" else ""))

    fig, ax = plt.subplots(figsize=poster_style.figsize(11.0, 6.0))
    # bottom was 0.30 to hold four lines of printed caption. With the caption
    # moved to the manuscript the axes get that space back.
    fig.subplots_adjust(top=0.90, bottom=0.145, left=0.105, right=0.985)

    # >>> THE SHADED BAND AND THE 0.5 LINE ARE DRAWN BUT NOT LABELLED.
    # Both used to carry grey on-axes text ("equilibration", "half consumed").
    # The caption explains them instead, which keeps the only text on the plot
    # the text a reader has to have: the axes and the key.
    # >>> THE BAND IS DRAWN AT THE TRUE EQUILIBRATION LENGTH.
    # r["eq_ps"] is the strided count and rounds UP to a whole strided frame:
    # 81 frames at 0.025 ps is 2.025 ps, but 9 strided frames at 0.25 ps is
    # 2.25 ps, so at --stride 10 the band was drawn a fifth of a picosecond too
    # wide. The curve still needs the strided value for its x-axis (see the
    # note at eq_ps), but the shading is a statement about the protocol and
    # must use the protocol's number.
    eq_max = max(r.get("eq_ps_true", r["eq_ps"]) for r in runs)
    if eq_max > 0:
        ax.axvspan(a.tmin, eq_max, color="0.85", alpha=.55, zorder=0)

    for r in sorted(runs, key=lambda r: -r["t_end"]):
        col, lab = classify(r["name"])
        m = r["t"] >= a.tmin
        ax.plot(r["t"][m], r["y"][m], color=col, lw=3.0, zorder=5,
                # publication legend: model name only; run lengths and the
                # failure time are in the caption (the end dot still marks it)
                label=lab)
        # >>> MARK THE END OF EVERY CURVE. A line that simply stops reads as a
        # plotting error; a line that stops at a dot reads as a death.
        if r["finished"]:
            ax.plot([r["t"][-1]], [r["y"][-1]], "o", color=col, ms=9,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=6)
        else:
            # an open arrow, not a filled dot: this run has not ended
            ax.plot([r["t"][-1]], [r["y"][-1]], ">", color="white", ms=11,
                    markeredgecolor=col, markeredgewidth=2.2, zorder=6)

    ax.axhline(0.5, color="0.55", ls="--", lw=1.2, zorder=1)

    hi = a.tmax if a.tmax else max(r["t_end"] for r in runs)
    if a.linear_time:
        ax.set_xlim(0, hi * 1.03)
        xlab = "time from start of equilibration (ps)"
    else:
        ax.set_xscale("log")
        ax.set_xlim(a.tmin, hi * 1.15)
        xlab = "time from start of equilibration (ps, log scale)"
    # >>> A CROPPED AXIS MUST NOT LOOK LIKE A RUN THAT ENDED.
    # The end-of-run dot is this figure's symbol for a death. If the axis is
    # cropped before a run finishes, that dot is off-screen and the curve just
    # runs into the spine, which is the one thing the marker exists to prevent
    # being ambiguous -- so say it on the figure.
    if a.tmax and any(r["t_end"] > a.tmax for r in runs):
        ax.text(0.99, 0.02, f"axis cropped at {a.tmax:g} ps; "
                f"all runs continue to "
                f"{max(r['t_end'] for r in runs):.0f} ps",
                transform=ax.transAxes, fontsize=9, style="italic",
                color="0.4", ha="right", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel("nitromethane remaining, per molecule", fontsize=11)
    # >>> NO ON-FIGURE TITLE BY DEFAULT.
    # The journal sets the figure number and title in the caption; a title in
    # the image duplicates it and cannot be edited at proof. --title restores
    # one for slides.
    if a.title:
        ax.set_title(a.title, loc="left", fontweight="bold", fontsize=12.5)
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=14, loc="lower left", framealpha=.95, handlelength=2.2)

    # >>> THE CAPTION TEXT IS NO LONGER ON THE FIGURE.
    # It was four italic grey lines under the axes. Everything it said belongs
    # in the manuscript caption, where it can wrap, be edited without rerunning
    # the script, and be typeset like the rest of the paper. What the script
    # still owes is the NUMBERS that caption has to state, so it prints them.
    print("\n  CAPTION MUST STATE (measured from the runs just plotted):")
    print(f"    density        {rho_label}")
    print(f"    connectivity   {a.scale}x the summed covalent radii")
    print(f"    smoothing      reactant rolled over {REACT_ROLL} ps")
    eqt = max(r.get("eq_ps_true", r["eq_ps"]) for r in runs)
    print(f"    shaded band    equilibration, to {eqt:.3f} ps "
          f"(the longest among these runs)")
    if abs(eqt - eq_max) > 1e-9:
        print(f"                   [band drawn to {eq_max:.3f} ps: the strided "
              f"grid cannot land exactly on\n                   the "
              f"equilibration boundary. Quote {eqt:.3f}. Half-lives are "
              f"unaffected.]")
    print( "    dashed line    half the initial nitromethane")
    print( "    line ends      a curve stops where its run stopped; the dot "
           "is its last frame")
    for r in sorted(runs, key=lambda r: -r["t_end"]):
        print(f"    {r['name']:<24} {r['rho']:.3f} g/cc, "
              f"{r['frame_ps']*1000/a.stride:.0f} fs/frame, "
              f"eq {r['eq_ps']:.1f} ps, ends {r['t_end']:.1f} ps")

    unfinished = [r["name"] for r in runs if not r["finished"]]
    if unfinished:
        print(f"\n  !! NOT FINISHED (no final.xyz): {', '.join(unfinished)}")
        print("     Drawn with an open arrow rather than a dot. Do NOT put this "
              "figure in the\n     manuscript until those runs complete -- a "
              "partial trajectory is indistinguishable\n     from a stability "
              "failure once it is a line on a plot.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"\n  wrote {a.out}")

    res = max(r.get("half_life_res_ps", 0) for r in runs)
    print(f"\n  half-life measured at full time resolution "
          f"({res*1000:.0f} fs), not at the plotting stride")
    print(f"\n  {'run':<24}{'t(1/2) ps':>11}{'at prod start':>15}{'ends ps':>10}")
    for r in sorted(runs, key=lambda r: r["t_half"] if np.isfinite(r["t_half"])
                    else 1e9):
        print(f"  {r['name']:<24}{r['t_half']:>11.2f}{r['y_at_prod']:>15.2f}"
              f"{r['t_end']:>10.1f}")


if __name__ == "__main__":
    main()
