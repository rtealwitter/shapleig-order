#!/bin/bash
# Direct before/after timing on a real HybridPairedEIG smoke run,
# including the readout path -- see profile_akzza_readout_compare.py's
# docstring for why the MSE validation's total_selection_time metric
# under-counted the compute_AKZZA_fast speedup.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-profile-akzza-readout
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/profile_akzza_readout_compare_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting AKZZA readout-path before/after comparison"
$PY ../profile_akzza_readout_compare.py
echo "[$(date)] rc=$?"
