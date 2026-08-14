#!/bin/bash
# Diagnostic: instruments EIGFunctionProperty.__call__'s sub-steps with
# timing wraps on a real resnet_14 HybridPairedEIG run, to find what's
# actually dominating acq_fun_duration after two targeted Cholesky fixes
# both measured safe but gave far less real speedup than expected.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-profile-acqfun
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/profile_acqfun_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting fastfit profiling"
$PY ../profile_acq_fun.py
echo "[$(date)] rc=$?"
