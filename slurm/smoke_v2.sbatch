#!/bin/bash
# Smoke test of the restructured refit logic + new games: two tiny sweeps —
# (a) dvbsgb_10, 5 arms x 2 seeds x 48 iterations (exhaustive candidates,
#     enough iterations to cross the hybrid's extremes->EIG handover), and
# (b) vit_16, 5 arms x 1 seed x 40 iterations (Subset-1024 candidates, to
#     exercise the size-fallback schedule replay and p=16 scalable mode).
# Requires the paired schedules (gen_schedules job) to exist.
#SBATCH --partition=debug
#SBATCH --account=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:59:00
#SBATCH --job-name=shapleig-smoke-v2
#SBATCH --output=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/run_logs/smoke_v2_%j.out

source ~/.bashrc
cd /hopper/groups/witterlab/rwitter/scratch/shapleig-order/shapleig-repo
PY=/hopper/groups/witterlab/rwitter/scratch/shapleig-order/.venv-repro/bin/python
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export MPLBACKEND=Agg

echo "[$(date)] node $(hostname) smoke (a): dv10 all arms"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_smoke_dv10
echo "[$(date)] smoke (a) rc=$?"
roota=$(ls -td multirun/*/* | head -1)

echo "[$(date)] smoke (b): vit_16 subset mode"
$PY -m xac.experiments.cli --config-path conf --config-name repro_all_smoke_p16
echo "[$(date)] smoke (b) rc=$?"
rootb=$(ls -td multirun/*/* | head -1)

echo "[$(date)] smoke roots: $roota $rootb"
$PY - "$roota" "$rootb" <<'EOF'
import glob, json, sys, os
for root in sys.argv[1:]:
    n = 0
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        met = json.load(open(mpath))
        n += 1
        acq = met.get("acquisition")
        mse = met.get("mse") or []
        hp = met.get("hp_fit_duration") or []
        ev = met.get("eval_fit_duration") or []
        aq = met.get("acq_fun_duration") or []
        sz = met.get("archive_sizes") or []
        print(f"{root.split('/')[-1]} {acq:28s} mse[{len(mse)}] last={mse[-1] if mse else None} "
              f"fits(method)={sum(1 for t in hp if t > 0)} fits(eval)={sum(1 for t in ev if t > 0)} "
              f"acq[{len(aq)}] sizes[{len(sz)}]")
    print(f"{root}: {n} runs")
EOF
echo "[$(date)] smoke summary done"
