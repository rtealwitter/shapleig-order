#!/bin/bash
# Aggregate sweep_roots_v2.txt (ShaplEIG, PairedExtremes[+fast],
# GP+Leverage, GP+Random -- unaffected by the Hybrid consolidation) plus
# sweep_roots_hybrid_akzza.txt (the single, AKZZA-accelerated Hybrid arm
# that replaces both old Hybrid variants) and regenerate the per-game
# figures. repro_plots.py's collect() drops any stale Hybrid run from the
# v2 roots by config field, so passing both root sets together is safe.
# Safe to rerun any time.
#SBATCH --partition=main
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --job-name=shapleig-aggregate-v3
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/agg_v3_%j.out

source ~/.bashrc
D=/hopper/groups/witterlab/rwitter/scratch/shapleig-order
PY=$D/.venv-repro/bin/python
export MPLBACKEND=Agg

roots=$(cat $D/sweep_roots_v2.txt $D/sweep_roots_hybrid_akzza.txt | sort -u | tr '\n' ' ')
echo "[$(date)] aggregating roots: $roots"
$PY $D/repro_plots.py $roots
echo "[$(date)] done rc=$?"
