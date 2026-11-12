#!/bin/bash
#SBATCH --job-name=orca_matrix
#SBATCH --partition=nielsen
#SBATCH --array=0-13
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%A_%a.log
#
# One Slurm array task per species: optimisation + frequencies, then the
# DLPNO-CCSD(T) single point on the optimised geometry. Serial ORCA (no %PAL
# in the inputs), so each task takes one core; the 14 species run side by
# side instead of one after another. Resumable: a finished job is skipped.
#
#   cd /mfs/io/groups/nielsen/Cary/MLIP/orca_matrix     (wherever this dir is)
#   sbatch submit_matrix.sh
#   squeue -u cxd190027
#   grep -L "TERMINATED NORMALLY" *_sp.out        # after it finishes
#
# --mem: %maxcore 8000 in the inputs is 8 GB per core, plus ORCA's own
# overhead, so 10 GB. Do not lower maxcore without also lowering --mem.

source "$HOME/orca_env.sh"
# Inside a Slurm job $0 is Slurm's private copy of this script, so dirname $0
# is an unwritable spool directory. Work where sbatch was invoked instead.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

SPECIES=(CH2OH CH2NH2 CH2NO2 CH2OOH CH2CH2OH CH3CH2O CH2CH2NH2 CH2CH2NO2
         CH2CH2OOH CH3CH2NH CH3CH2NO CH3CH2OH CH3CH2NO2 CH3CH2OOH)
name=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ok () { grep -q "ORCA TERMINATED NORMALLY" "$1" 2>/dev/null; }
run () {
    ok "$1.out" && { echo "skip $1"; return 0; }
    echo "=== $1  $(date)  on $(hostname)"
    "$ORCA_BIN" "$1.inp" > "$1.out" 2>&1
    ok "$1.out" || { echo "    FAILED: see $1.out"; return 1; }
}

run "${name}_opt" || exit 1
[ -s "${name}_opt.xyz" ] || { echo "no ${name}_opt.xyz; not running single point"; exit 1; }
run "${name}_sp"
echo "done $name $(date)"
