# mlip-nitromethane-benchmark

Data and scripts for

> C. B. Darwin, J. Dunal, S. O. Nielsen, "[paper title]", *J. Chem. Theory Comput.* [year, volume, pages, DOI].

The paper benchmarks five machine-learned interatomic potentials (MACE-OFF23,
MACE-Osaka24, MACE-OMol-0, UMA-S-1P1, ANI-1xnr) and a fine-tuned ANI-1xnr
(ANI-1xnr-Mrr) on the decomposition of condensed-phase nitromethane.

The layout mirrors the working directory used for the paper, so the scripts run
unmodified from the repository root. Molecular dynamics trajectories are too
large for GitHub and are archived on Zenodo: [Zenodo DOI].

## Requirements

Two separate Python environments are needed, because the MACE and fairchem
(UMA) stacks cannot be loaded in the same process.

| Environment | Used for | Packages |
|---|---|---|
| `mace-dox` | MACE-OFF23, MACE-Osaka24, MACE-OMol-0, ANI-1xnr | ASE [version], mace-torch [version], torchani [version], PyTorch [version] |
| `uma` | UMA-S-1P1 | ASE [version], fairchem-core [version], PyTorch [version] |

Reference calculations used ORCA 6.1.1. Model weights are not redistributed
here, except the fine-tuned ANI-1xnr-Mrr ensemble. Obtain the others from their
original sources (see the Supporting Information, Table S1).

## Contents

| Path | What it is |
|---|---|
| `nm_box_1p97_clean.xyz`, `nm_box_1p97_clean_crystal.xyz` | Starting cells for all liquid and crystal runs (1.970 g/cm3) |
| `build_nm_box.py`, `add_methyl_h.py`, `files/prep_dense_box.py`, `check_box.py` | Construction and checking of the starting cells |
| `run_reactive_md.py`, `reactive_lowdens.py`, `zbl_patch.py` | Molecular dynamics driver and the ZBL repulsion |
| `runs/` | Logs and run manifests of the production runs (trajectories on Zenodo) |
| `species_analysis.py` | Species assignment from saved frames |
| `files/bond_pull_scan.py`, `seam_scan/` | Rigid dissociation scans and their results |
| `files/dft/dft_full/` | ORCA inputs, outputs and energies of the DFT C-N scan (RKS and broken-symmetry UKS) |
| `orca_molecules.py`, `orca_dlpno/`, `orca_mol/` | Reference calculations for the 26-bond set |
| `orca_matrix.py`, `orca_matrix/`, `matrix_geoms/`, `matrix_models.py`, `matrix_models/` | Reference calculations and model single points for the 17-cell matrix |
| `finetune_ani_morse.py`, `ANI-1xnr-Mrr/` | Fine-tuning script and the eight fine-tuned ensemble members |
| `short_range_floor.py`, `short_range_floor/` | Compression test |
| `ablate_embedding.py`, `ablate_subset.py`, `falsify_label.py` | Ablation, control and falsification tests |
| `nve_timestep_test.py` | Time-step test for the isolated molecule |
| `MLIP_paper/*.csv` | Tabulated results read by the plotting scripts |

## Figures

| Figure | Script | Main input |
|---|---|---|
| 1 | `make_comparison_figure.py` | `seam_scan/pull_*.csv`, `files/dft/dft_full/dft_energies_full.csv` |
| 2 | `make_model_comparison_matrix.py` | `bde_matrix_data.py` (from `matrix_models.py --merge`) |
| 3, 4 | `plot_reactant_survival.py` | trajectories (Zenodo) |
| 5 | `plot_short_range_floor.py` | `short_range_floor/floor_*.csv` |
| 6 | `plot_zbl_split.py` | trajectories (Zenodo) |
| 7 | `plot_mrr_remediation.py` | `MLIP_paper/*.csv` |
| 8 | `plot_mrr_products.py` | trajectories (Zenodo) |
| 9 | `plot_nve_dynamics.py` | isolated-molecule trajectory (Zenodo) |
| 10 | `plot_tasks.py` | `seam_scan/pull_umaTASK_*.csv`, `MLIP_paper/task_label_test.csv` |
| 11 | `plot_ablation_scans.py` | `MLIP_paper/ablation_*_ladder_*.csv` |
| 12 | `plot_falsify.py` | `MLIP_paper/falsify_label_*.csv` |

## Notes

- The crystal structure of nitromethane (CSD refcode NTROMA, CCDC 1223555) is
  not redistributed. Obtain it from the CCDC and run `add_methyl_h.py`.
- The crystal starting cell contains 71 intact molecules, one CH2NO2 fragment
  and one detached hydrogen atom. See the Supporting Information, Section S5.
- Some scripts contain cluster-specific paths. Adjust them to your system.

## License

[choose a license, for example MIT for the scripts and CC BY 4.0 for the data]
