# ANI-1xnr-Mrr

Fine-tuned ANI-1xnr. **M** = Morse (PCDA), **rr** = radical-radical targeted.

## Usage

`ani_1xnr_Mrr.py` lives in this folder and is not installed as a package, so put
this folder on `sys.path` (or `cd` into it) before importing:

    import sys
    sys.path.insert(0, "MLIP-Benchmark-BDE-Heatmap/ANI-1xnr-Mrr")
    from ani_1xnr_Mrr import get_calculator

    calc = get_calculator(info="ani-1xnr/model/ani-1xnr.info")
    atoms.calc = calc

The loader locates `member0.pt` .. `member7.pt` relative to its own file, so once
imported the working directory is irrelevant. `info=` must point at the **original
ANI-1xnr release**, which supplies the architecture and AEV parameters -- the
`.pt` files here are state dicts only and are useless without it.

Requires `torchani` and the original ANI-1xnr release (the `ani-1xnr/` directory
containing `train0` .. `train7`). All eight members are loaded and averaged, as
in the released model, via a shared-AEV torchani Ensemble (~1.6x the cost of one
member, not 8x). Per-member spread is available from `energy_spread()` but is
**not** a valid error bar for this model -- see LIMITATIONS.

## What was done

Physics-constrained data augmentation following dos Santos et al.,
*J. Chem. Inf. Model.* **2025**, 65, 1198-1210, with two deviations:

1. **Energy-only labelling of the Morse frames**, the fallback their limitations
   section suggests, rather than the reference-MLIP forces used in the paper.
   Force information instead comes from a rehearsal ("anchor") set labelled by
   the frozen original model.
2. **Targeted at radical-radical fragmentations** -- bonds where both fragments
   are polyatomic. Only 2 of the ~43 bonds in their species set are of this type
   (H2O2 O-O and HOCO C-O), and it is where the BDE error reverses sign.

Reference dissociation energies are our own DLPNO-CCSD(T)=FULL/aug-cc-pVQZ
//wB97X-D3/def2-TZVP calculations (16 bonds canonical CCSD(T), 10 DLPNO). D0
values reproduce experiment to 0.06 kcal/mol for methane and 0.4 for
nitromethane.

## Result

12 bonds augmented, 26 evaluated. All numbers below are for the **8-member
ensemble average** -- the thing this package actually predicts. Errors are
prediction minus DLPNO-CCSD(T) De, in kcal/mol; relaxed geometries.

| class | before | after |
| --- | --- | --- |
| radical-radical, mean abs. error | 28.3 | **4.8** |
| atom-producing, mean abs. error  | 14.3 | 14.6 |
| CH3NO2 C-N                       | -48.2 | **-0.4** |
| NO2 N-O                          | +15.8 | **-23.9** |

Radical-radical bonds, per bond:

| bond | before | after | member spread (after) |
| --- | --- | --- | --- |
| CH3NO2 C-N    | -48.2 | -0.4 | +/- 2.6 |
| CH3OH C-O     | -12.8 | +0.6 | +/- 1.3 |
| H2O2 O-O      |  -3.8 | +0.8 | +/- 1.0 |
| N2H4 N-N      | -23.5 | +1.3 | +/- 0.8 |
| CH3OCH3 C-O   | -42.9 | +4.8 | +/- 2.4 |
| C2H6 C-C      | -17.3 | +5.0 | +/- 2.3 |
| CH3ONO C-O    | -42.2 | -5.0 | +/- 1.6 |
| CH3CH2NH2 C-N | -27.7 | -6.9 | +/- 1.6 |
| CH3COOH C-C   | -36.7 | -18.5 | +/- 3.5 |

Note the residual is structured, not scatter: C-C and C-O-C overshoot by ~+5,
the N- and COOH-containing bonds undershoot. A least-squares fit over per-species
energy shifts predicts a residual of this kind, so it is probably the method's
floor rather than undertraining.

## LIMITATIONS -- read before using

* **Corrected on 12 bonds, validated on 26.** This is not a general-purpose
  improvement to ANI-1xnr and should not be described as one.

* **THE ENSEMBLE SPREAD IS NOT AN ERROR BAR.** All eight members were fine-tuned on the
  same augmented data, so their agreement is induced rather than informative.
  On the atom-producing bonds the error did not move while the spread collapsed:

  | bond | error before -> after | spread before -> after |
  | --- | --- | --- |
  | N2 | -6.9 -> -6.9 | +/-55.7 -> +/-1.4 |
  | CN | +22.3 -> +22.8 | +/-32.7 -> +/-1.7 |
  | NO | +21.7 -> +21.7 | +/-25.0 -> +/-1.8 |

  Same error, ~30x more confidence. Do not shade a figure by this quantity and
  present it as accuracy. `energy_spread()` will compute it; resist the urge.

* **Atom-producing bonds did not improve** (ensemble MAE 14.3 -> 14.6). The
  per-member MAE does fall (18.1 -> 14.6), but only because the original members
  disagreed wildly and their errors partly cancelled in the average; fine-tuning
  removed the disagreement and with it the cancellation. The honest statement is
  that this correction targets radical-radical bonds and leaves atom-producing
  bonds where it found them.

* **NO2 is WORSE than the original**: its own N-O BDE went +15.8 -> -23.9
  kcal/mol, consistently across all 8 members. NO2 is simultaneously a fragment
  whose energy had to rise ~50 kcal/mol (to fix CH3NO2) and a parent with its own
  augmented curve; those demands are incompatible. Treat NO2, and any bond whose
  fragments include NO2, as unreliable.

* **CH3COOH C-C fails only on relaxation.** At fixed reference geometry its error
  is +9.5 kcal/mol (fine); after relaxing the fragments it is -18.5. The gap was
  -9.4 kcal/mol before fine-tuning and is -28.1 after, so the augmentation
  deepened a spurious off-data minimum in the fragments rather than creating one.
  Any bond producing COOH should be spot-checked against its vertical value.

* Individual atom-producing bonds moved by 10-40 kcal/mol as collateral (NO2,
  H2, OH, NH3, N2). The correction propagates through shared per-element
  networks, so species that were never trained can shift.

* MD behaviour is **untested** at the time of export.

## Note on the original ensemble

Member 6 of the released ANI-1xnr is a large outlier on the triple-bonded and
radical species (N2 -133.0, CN -53.4, NO -34.2 kcal/mol error) and member 1 is a
milder one. This is a property of the released model, not of the fine-tuning, and
it is the main source of the large "before" spreads above. Fine-tuning happened
to repair it (member 6 after: N2 -7.2, CN +22.9, NO +25.1).

## Files

    member0.pt .. member7.pt   fine-tuned state dicts, one per ensemble member
    ani_1xnr_Mrr.py            loader -> ASE calculator
    provenance.json            recipe, hyperparameters, results
