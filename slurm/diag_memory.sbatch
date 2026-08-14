#!/bin/bash
# Single-process, single-worker memory diagnostic: repeatedly calls
# AcceleratedFitConfig's fit_accelerated on a growing archive (t up to 300)
# to see whether ru_maxrss grows unboundedly across many sequential fits --
# isolating this from 90-way joblib parallelism, which the two OOM'd full
# sweeps couldn't distinguish from a per-call memory spike.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-diag-mem
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/diag_mem_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting memory diagnostic"
$PY diag_memory.py
echo "[$(date)] rc=$?"
