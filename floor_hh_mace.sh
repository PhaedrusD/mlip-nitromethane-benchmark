#!/bin/bash
#SBATCH --job-name=floor_hh_mace
#SBATCH --output=floor_hh_mace_%j.out
#SBATCH --error=floor_hh_mace_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=nielsen
#SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
#SBATCH --time=01:00:00
#
# H-H compression cut for the four models that live in the mace-dox
# environment. One python process per model. UMA runs in floor_hh_uma.sh,
# in its own job, because the two environments cannot share a shell.
# Output: MLIP_paper/short_range_floor_hh/floor_<model>.csv
source ~/conda-envs/activate_mace.sh
export TORCHANI_NO_WARN_EXTENSIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /mfs/io/groups/nielsen/Cary/MLIP

FRAME=osaka_run_2/production.traj     # same frame as the O-H cut (index 2000)
python short_range_floor.py --pair HH --model off23   --frame $FRAME --index 2000 --device cuda
python short_range_floor.py --pair HH --model osaka24 --frame $FRAME --index 2000 --device cuda \
    --osaka-path /mfs/io/groups/nielsen/Cary/MLIP/mace-osaka24-medium-headfix.model
python short_range_floor.py --pair HH --model omol    --frame $FRAME --index 2000 --device cuda
python short_range_floor.py --pair HH --model ani     --frame $FRAME --index 2000 --device cuda \
    --ani-info /mfs/io/groups/nielsen/Cary/MLIP/ani-1xnr/model/ani-1xnr.info
ls -la MLIP_paper/short_range_floor_hh/
