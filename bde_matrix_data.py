"""
bde_matrix_data.py -- the 17-cell bond x environment error matrix, one copy
============================================================================

>>> REFERENCE (changed 2026-10-22)
    Errors are model minus DLPNO-CCSD(T)=FULL/aug-cc-pVQZ De, both evaluated
    as single points at the wB97X-D3/def2-TZVP geometries (orca_dlpno/,
    orca_matrix/; model energies from matrix_models.py). The previous
    version scored BFGS-relaxed model fragments against hand-typed
    experimental D(298 K) values of unrecorded origin; it is kept as
    bde_matrix_data_exp298.py. Two of its cells were artefacts: the
    peroxide C-C (the CH2OOH fragment fell apart under relaxation) and the
    nitro N-O (reference 59.0, should be ~97; OMOL/UMA run with the wrong
    multiplicities). Regenerate from matrix_models.py --merge, never by hand.

>>> WHY THIS FILE EXISTS
    These numbers were typed into make_bubble_matrix.py as five literal arrays.
    The moment a SECOND figure wants them, a copy exists, and a copy of a
    hardcoded table is the kind of error nobody catches: the two figures drift,
    both look right, and only a reader adding them up finds the disagreement.
    One module, imported by every figure that draws these values.

>>> THESE ARE NOT READ FROM THE JSON
    make_bubble_matrix.py's docstring claims it "uses the results json file".
    It does not, and has not for some time -- the values below were transcribed
    by hand from the thermodynamic BDE run. That is fine for a frozen result,
    but it means a rerun of thermodynamic_bde_calculation.py does NOT propagate
    here. If the benchmark is rerun, these numbers must be retranscribed, and
    the retranscription is the step to check.

>>> SIGN CONVENTION
    Positive = the model predicts a STRONGER bond than the reference.
    Negative = weaker. ANI-1xnr's -60.5 on nitro C-N is the number behind the
    box destroying itself during a 300 K equilibration.

>>> THE GRID IS 42 CELLS AND 17 ARE FILLED
    A bond only exists in an environment that has it: there is no N-O bond in
    an alkane. The NaNs are structural absences, not missing measurements, and
    a figure must not imply otherwise.
"""

import numpy as np

BONDS = ["C-H", "C-C", "C-O", "N-H", "C-N", "N-O", "O-O"]
ENVIRONMENTS = ["Alkanes", "Alcohols", "Amines", "Nitro\nCompounds",
                "Peroxides", "Carbonyls"]

data_mace_off23 = np.array([
    [   8.03,    7.95,    8.02,    5.16,    7.97,   -3.04],  # C-H
    [  18.59,   15.57,   10.46,    5.01,    8.65, np.nan],  # C-C
    [np.nan,  -22.99, np.nan, np.nan, np.nan, np.nan],  # C-O
    [np.nan, np.nan,    5.47, np.nan, np.nan, np.nan],  # N-H
    [np.nan, np.nan,   11.35,   20.58, np.nan, np.nan],  # C-N
    [np.nan, np.nan, np.nan,   -0.83, np.nan, np.nan],  # N-O
    [np.nan, np.nan, np.nan, np.nan,  -33.95, np.nan],  # O-O
])
data_mace_osaka24 = np.array([
    [  -1.96,   -3.33,   -3.15,   -3.95,   -2.28,  -13.87],  # C-H
    [   5.73,    3.53,   -0.73,   -6.34,   -0.41, np.nan],  # C-C
    [np.nan,   -4.30, np.nan, np.nan, np.nan, np.nan],  # C-O
    [np.nan, np.nan,   -0.57, np.nan, np.nan, np.nan],  # N-H
    [np.nan, np.nan,    4.44,  -12.88, np.nan, np.nan],  # C-N
    [np.nan, np.nan, np.nan,   -5.84, np.nan, np.nan],  # N-O
    [np.nan, np.nan, np.nan, np.nan,  -14.06, np.nan],  # O-O
])
data_ani_1xnr = np.array([
    [  -8.62,   -5.14,   -5.29,  -11.70,   -5.59,   -2.00],  # C-H
    [ -17.49,  -37.28,  -38.37,  -47.10,  -44.60, np.nan],  # C-C
    [np.nan,  -30.51, np.nan, np.nan, np.nan, np.nan],  # C-O
    [np.nan, np.nan,  -10.67, np.nan, np.nan, np.nan],  # N-H
    [np.nan, np.nan,  -28.50,  -72.18, np.nan, np.nan],  # C-N
    [np.nan, np.nan, np.nan,  -19.86, np.nan, np.nan],  # N-O
    [np.nan, np.nan, np.nan, np.nan,  -29.21, np.nan],  # O-O
])
data_mace_omol = np.array([
    [  -4.77,   -3.70,   -3.44,   -4.43,   -3.56,   -4.81],  # C-H
    [  -7.01,   -6.38,   -6.26,   -6.24,   -5.23, np.nan],  # C-C
    [np.nan,   -2.87, np.nan, np.nan, np.nan, np.nan],  # C-O
    [np.nan, np.nan,   -3.63, np.nan, np.nan, np.nan],  # N-H
    [np.nan, np.nan,   -3.34,   -4.43, np.nan, np.nan],  # C-N
    [np.nan, np.nan, np.nan,    4.41, np.nan, np.nan],  # N-O
    [np.nan, np.nan, np.nan, np.nan,   -1.95, np.nan],  # O-O
])
data_uma_s_1p1 = np.array([
    [  -1.16,   -1.21,   -1.10,   -1.26,   -1.03,   -2.28],  # C-H
    [  -1.11,   -2.09,   -2.25,   -2.15,   -2.13, np.nan],  # C-C
    [np.nan,   -0.69, np.nan, np.nan, np.nan, np.nan],  # C-O
    [np.nan, np.nan,   -1.33, np.nan, np.nan, np.nan],  # N-H
    [np.nan, np.nan,   -0.25,   -1.03, np.nan, np.nan],  # C-N
    [np.nan, np.nan, np.nan,    0.87, np.nan, np.nan],  # N-O
    [np.nan, np.nan, np.nan, np.nan,   -1.47, np.nan],  # O-O
])

DATA = {
    "MACE-OFF23":   data_mace_off23,
    "MACE-Osaka24": data_mace_osaka24,
    "ANI-1xnr":     data_ani_1xnr,
    "MACE-OMOL":    data_mace_omol,
    "UMA-S-1P1":    data_uma_s_1p1,
    # >>> ANI-1xnr-Mrr IS NOT HERE ON PURPOSE.
    # The remediated model has never been run through the full thermodynamic
    # BDE benchmark that fills these 17 cells -- only through the 26-bond
    # dissociation table in MLIP_paper/mrr_bde_table.csv, which covers a
    # different and smaller set of species. Pasting the class means from that
    # table into this grid would put two different benchmarks side by side
    # under one colour bar. To add the column properly, run
    # thermodynamic_bde_calculation.py with the ANI-1xnr-Mrr checkpoints.
}

# >>> ORDER IS AN ARGUMENT, NOT AN ALPHABET.
# Left to right: the two models that destroy their own box in MD, then the
# reactive model, then the two that survive. A reader who knows nothing else
# about the paper can read the MD outcome off the column order.
MODEL_ORDER = ["MACE-OFF23", "MACE-Osaka24", "ANI-1xnr", "MACE-OMOL",
               "UMA-S-1P1"]

# >>> ONE PALETTE FOR THE WHOLE PAPER.
# plot_reactant_survival.py and make_comparison_figure.py currently disagree
# about ANI-1xnr: teal in one, red in the other, while ANI-1xnr-Mrr is teal in
# the first. A model that changes colour between figures is read as two models.
# This is the survival palette, because it is the only one with room for both
# ANI variants.
COLORS = {
    "ANI-1xnr":     "#b03030",
    "ANI-1xnr-Mrr": "#0f7d7d",
    "MACE-OFF23":   "#e8860c",
    "MACE-Osaka24": "#2e8b30",
    "MACE-OMOL":    "#1f4fd8",
    "UMA-S-1P1":    "#8e6bbf",
}


def cells(models=None):
    """The 17 populated cells as rows.

    Returns a list of dicts with the bond, the environment, a one-line label,
    and the per-model error vector in `models` order. Cells are located by
    scanning the grid, so a cell that is filled for one model and empty for
    another would be caught rather than silently dropped -- it does not happen
    in this data (all five models cover the same 17), and the assertion below
    is what keeps that true if the table is ever extended.
    """
    models = list(models or MODEL_ORDER)
    stack = np.array([DATA[m] for m in models])          # (n_model, 7, 6)
    filled = ~np.isnan(stack)

    out = []
    for i, b in enumerate(BONDS):
        for j, e in enumerate(ENVIRONMENTS):
            col = filled[:, i, j]
            if not col.any():
                continue
            assert col.all(), (
                f"cell {b} / {e} is present for some models and absent for "
                f"others: {[m for m, f in zip(models, col) if not f]}. "
                "A partially populated cell cannot go on a shared grid.")
            v = stack[:, i, j]
            out.append(dict(bond=b, env=e.replace("\n", " "),
                            label=f"{b.replace('-', '–')}  ·  "
                                  f"{e.replace(chr(10), ' ')}",
                            values=v,
                            spread=float(v.max() - v.min()),
                            maxabs=float(np.abs(v).max())))
    return out
