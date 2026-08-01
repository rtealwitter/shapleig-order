#!/bin/bash
# Smoke test of the shapleig replication: generate the paired schedule for
# p=10, then run 8 tiny hydra jobs (1 game x 4 acquisitions x 2 seeds x 8
# iterations) to flush config/import errors before the full sweep.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:50:00
#SBATCH --job-name=shapleig-hybrid-smoke
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/hybrid_smoke_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

echo "[$(date)] node $(hostname)"
if [ ! -f data/paired_schedule_p10.npy ]; then
  $PY - <<'EOF'
import sys
sys.path.insert(0, "/hopper/groups/witterlab/rwitter/scratch/shapleig-order/greedy")
import numpy as np
from gp_core import fixed_schedule
s = fixed_schedule(10, 512)
np.save("data/paired_schedule_p10.npy", s)
sizes = [bin(int(m)).count("1") for m in s[:26]]
print("schedule head sizes:", sizes)
EOF
fi
echo "[$(date)] schedule ready; running smoke sweep"
$PY -m xac.experiments.cli --config-path conf --config-name repro_dv10_hybrid_smoke
echo "[$(date)] smoke done rc=$?"
find multirun -name metrics.json | head -10
