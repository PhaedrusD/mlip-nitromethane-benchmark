#!/bin/bash
#SBATCH --job-name=floor_hh_uma
#SBATCH --output=floor_hh_uma_%j.out
#SBATCH --error=floor_hh_uma_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=nielsen
#SBATCH --gres=gpu:tesla_v100-pcie-16gb:1
#SBATCH --time=01:00:00
#
# H-H compression cut for UMA-S-1P1, in the uma environment only.
# Output: MLIP_paper/short_range_floor_hh/floor_uma.csv
source "/opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh"
conda activate ~/conda-envs/uma
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /mfs/io/groups/nielsen/Cary/MLIP

python short_range_floor.py --pair HH --model uma --uma-path uma-s-1p1 \
    --frame osaka_run_2/production.traj --index 2000 --device cuda
ls -la MLIP_paper/short_range_floor_hh/
