#!/usr/bin/env python
"""
species_analysis.py
=====================================================================
Connectivity-based reaction analysis for reactive nitromethane MD,
producing Han et al. (JPCB 2011)-style diagnostics from any trajectory
(single-molecule OR low-density box), for any of the five MLIPs.

Two products of this module (both feed the paper reframe):
  1. species_populations(...)  -> Han Fig 5 : population per NM vs time
  2. first_bond_events(...)    -> Han Table 1: first break/formation times

Bonding is geometric: atoms i,j are bonded if
        r_ij < scale * (r_cov_i + r_cov_j)
with minimum-image convention when the cell is periodic. A break/formation
is only registered once it has PERSISTED for `debounce` frames, so a single
thermal excursion across the cutoff is not miscounted as chemistry -- this
matters enormously for first-passage statistics.

No dependency beyond numpy + ASE (connected components done with a local BFS
so there is no networkx/scipy requirement on the cluster analysis nodes).

Design notes for reuse:
  * `scale` ~ 1.2 defines a "bond present" for species assignment.
  * `break_scale` ~ 1.6 (looser) defines a bond as decisively broken; the gap
    between the two gives hysteresis so flickering near ~1.2 is not a "break".
  * All times are returned in ps given the per-frame dt (fs) and log stride.
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from ase.data import covalent_radii

# ---------------------------------------------------------------------------
# Friendly labels, keyed on Hill formula. Isomers share a formula, so
# nitromethane / methyl nitrite / aci-nitromethane all fall under CH3NO2.
# ---------------------------------------------------------------------------
SPECIES_LABELS = {
    "CH3NO2": "CH3NO2 (NM+isomers)",   # C1H3N1O2 : NM, CH3ONO, CH2NOOH
    "CH4NO2": "CH3NOOH (+H)",          # C1H4N1O2 : protonated intermediate
    "H2O":    "H2O",
    "CO2":    "CO2",
    "N2":     "N2",
    "H3N":    "NH3",
    "CH4":    "CH4",
}
# Order used for plotting / table columns (everything else -> "other").
SPECIES_ORDER = ["CH3NO2 (NM+isomers)", "CH3NOOH (+H)", "H2O", "CO2",
                 "N2", "NH3", "CH4", "other"]


def hill_formula(symbols) -> str:
    """Hill-system formula string for a list of element symbols."""
    c = Counter(symbols)
    def tok(el):
        return el + (str(c[el]) if c[el] > 1 else "")
    if "C" in c:
        parts = [tok("C")]
        if "H" in c:
            parts.append(tok("H"))
        parts += [tok(el) for el in sorted(k for k in c if k not in ("C", "H"))]
    else:
        parts = [tok(el) for el in sorted(c)]
    return "".join(parts)


def label_for(symbols) -> str:
    return SPECIES_LABELS.get(hill_formula(symbols), "other")


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def adjacency(atoms, scale=1.2, mic=None):
    """Boolean N x N bond matrix from a geometric covalent criterion."""
    if mic is None:
        mic = bool(atoms.pbc.any())
    D = atoms.get_all_distances(mic=mic)
    rcov = covalent_radii[atoms.numbers]
    thresh = scale * (rcov[:, None] + rcov[None, :])
    A = D < thresh
    np.fill_diagonal(A, False)
    return A


def connected_components(A):
    """List of index-lists for each connected component (iterative BFS)."""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp, seen[s] = [s], [s], True
        while stack:
            u = stack.pop()
            for v in np.nonzero(A[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    stack.append(v)
        comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
# Han Fig 5 : species populations vs time
# ---------------------------------------------------------------------------
def species_populations(traj, dt_fs, log_stride=1, scale=1.2,
                        roll_ps=2.0, n_nitromethane=None):
    """
    Parameters
    ----------
    traj : list[Atoms]        # ordered frames
    dt_fs : float             # integrator timestep in fs
    log_stride : int          # frames were written every `log_stride` steps
    roll_ps : float           # rolling-average window (Han uses 2 ps)
    n_nitromethane : int      # normaliser; default = species count in frame 0

    Returns
    -------
    t_ps : (F,) array
    pops : dict[label] -> (F,) array   # population per nitromethane
    """
    symbols = np.array(traj[0].get_chemical_symbols())
    if n_nitromethane is None:
        A0 = adjacency(traj[0], scale)
        n_nitromethane = sum(
            label_for(symbols[c]) == "CH3NO2 (NM+isomers)"
            for c in connected_components(A0)
        )
    n_nitromethane = max(int(n_nitromethane), 1)

    frame_dt_ps = dt_fs * log_stride / 1000.0
    F = len(traj)
    raw = {lab: np.zeros(F) for lab in SPECIES_ORDER}
    for f, atoms in enumerate(traj):
        A = adjacency(atoms, scale)
        counts = Counter(label_for(symbols[c]) for c in connected_components(A))
        for lab in SPECIES_ORDER:
            raw[lab][f] = counts.get(lab, 0)

    t_ps = np.arange(F) * frame_dt_ps
    win = max(int(round(roll_ps / frame_dt_ps)), 1)
    kern = np.ones(win)
    # Edge-correct: divide by the number of frames actually inside the window
    # at each position (else np.convolve 'same' halves the endpoints).
    counts = np.convolve(np.ones(F), kern, mode="same")
    pops = {lab: (np.convolve(v, kern, mode="same") / counts) / n_nitromethane
            for lab, v in raw.items()}
    return t_ps, pops


# ---------------------------------------------------------------------------
# Han Table 1 : first bond breaking / formation times
# ---------------------------------------------------------------------------
def _typed_pairs(atoms0, scale=1.2):
    """{('C','N'): [(i,j), ...]} for bonds present in the initial frame."""
    A = adjacency(atoms0, scale)
    sym = atoms0.get_chemical_symbols()
    out = {}
    iu, ju = np.triu(A, 1).nonzero()
    for i, j in zip(iu, ju):
        key = tuple(sorted((sym[i], sym[j])))
        out.setdefault(key, []).append((int(i), int(j)))
    return out


def _pair_distance_series(traj, pairs, mic):
    """(F, n_pairs) distance array for a fixed list of (i,j)."""
    F = len(traj)
    d = np.empty((F, len(pairs)))
    for f, atoms in enumerate(traj):
        for k, (i, j) in enumerate(pairs):
            d[f, k] = atoms.get_distance(i, j, mic=mic)
    return d


def _first_sustained(mask, debounce):
    """First index where `mask` is True for >= debounce consecutive frames."""
    run = 0
    for f, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= debounce:
            return f - debounce + 1
    return None


def first_bond_events(traj, dt_fs, log_stride=1, scale=1.2,
                      break_scale=1.6, form_scale=1.2, debounce_fs=20.0):
    """
    Returns two dicts of first-event times in ps (or None):
      breaks : bonds present at t0 that first exceed break_scale*(rcov+rcov)
      forms  : C-O and O-H pairs absent at t0 that first come within form_scale
    Debouncing requires the new state to persist `debounce_fs` before it counts.
    """
    frame_dt_ps = dt_fs * log_stride / 1000.0
    debounce = max(int(round(debounce_fs / (dt_fs * log_stride))), 1)
    mic = bool(traj[0].pbc.any())
    sym = np.array(traj[0].get_chemical_symbols())
    rcov = covalent_radii[traj[0].numbers]

    # ---- breaks (track bonds that exist initially) -------------------------
    typed = _typed_pairs(traj[0], scale)
    breaks = {}
    for key, pairs in typed.items():
        d = _pair_distance_series(traj, pairs, mic)
        first = None
        for k, (i, j) in enumerate(pairs):
            thr = break_scale * (rcov[i] + rcov[j])
            idx = _first_sustained(d[:, k] > thr, debounce)
            if idx is not None:
                first = idx if first is None else min(first, idx)
        breaks["-".join(key)] = None if first is None else first * frame_dt_ps

    # ---- formations (C-O, O-H not bonded at t0) ----------------------------
    A0 = adjacency(traj[0], scale)
    forms = {}
    for want in (("C", "O"), ("H", "O")):
        idxA = np.where(sym == want[0])[0]
        idxB = np.where(sym == want[1])[0]
        cand = [(int(i), int(j)) for i in idxA for j in idxB
                if i != j and not A0[i, j]]
        if not cand:
            forms["-".join(want)] = None
            continue
        d = _pair_distance_series(traj, cand, mic)
        first = None
        for k, (i, j) in enumerate(cand):
            thr = form_scale * (rcov[i] + rcov[j])
            idx = _first_sustained(d[:, k] < thr, debounce)
            if idx is not None:
                first = idx if first is None else min(first, idx)
        forms["-".join(want)] = None if first is None else first * frame_dt_ps

    return breaks, forms