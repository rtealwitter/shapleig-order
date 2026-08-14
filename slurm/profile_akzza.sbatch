#!/bin/bash
# Diagnostic: times exact vs. accelerated (current: fold+forced-iterative)
# vs. accelerated-fold-direct-Cholesky fits on real resnet_14/vit_16
# archives (t~500+, p=14/16), to test whether forcing CG/Lanczos on top of
# odd-part folding is itself the regression that makes AcceleratedFitConfig
# slower than the exact fit on these two games specifically.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-profile-akzza
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/profile_akzza_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting fastfit profiling"
$PY ../profile_akzza.py
echo "[$(date)] rc=$?"
