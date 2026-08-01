#!/usr/bin/env bash
# One-shot numerics: run the fixed-kernel greedy ShaplEIG ordering experiment
# on a compute node (small, CPU-only, minutes). Rerun by resubmitting; the
# script is stateless and just prints its analysis to the job log.
#SBATCH --job-name=shapleig-order
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --export=ALL
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/greedy_order_%j.out

source ~/.bashrc
echo "[$(date)] node $(hostname) python $(command -v python)"
python /hopper/groups/witterlab/rwitter/scratch/shapleig-order/greedy/greedy_order.py
echo "[$(date)] done rc=$?"
