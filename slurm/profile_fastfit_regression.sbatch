#!/bin/bash
# Instrumented before/after (exact vs AcceleratedFitConfig) on resnet_14,
# vit_9, and dvbsgb_10 (as a healthy control) at smoke scale, to find the
# mechanism behind the acq_fun_duration regression on resnet_14/vit_9 seen
# in the existing sweep data (sweep_roots_v2.txt).
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-profile-fastfit-regression
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/profile_fastfit_regression_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

echo "[$(date)] starting fast-fit regression investigation"
$PY ../profile_fastfit_regression.py
echo "[$(date)] rc=$?"
