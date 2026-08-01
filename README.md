# shapleig-order

The ShaplEIG selection rule [RFMBF, ICML '26] is value-independent given the
kernel hyperparameters: Gaussian conditioning changes the posterior covariance
only through *which* coalitions were evaluated, so with frozen lengthscales
the entire "adaptive" design is determined before the first evaluation. This
repository works out what that frozen-kernel design is (complement-paired
extremes, then balanced middles), replicates the paper's Figure 3 ablation on
its own games and code, and adds two arms: the fixed schedule replayed with no
adaptivity, and a hybrid that selects extremes without fitting and then
switches to EIG with refits on a geometric schedule. The writeup is
`shapleig_greedy_note.md`.

## Layout

- `shapleig_greedy_note.md` — the note: closed form for the frozen-kernel
  greedy, how the adaptive rule drifts along it, replication tables.
- `shapleig-repo/` — the authors' released code
  ([slds-lmu/shapleig](https://github.com/slds-lmu/shapleig), MIT), vendored
  at upstream commit `162ce44` with modifications; the diff against upstream
  is `upstream-changes.patch`, and the original git dir is preserved locally
  as `.git-upstream` (not pushed). Main changes:
  - `PairedExtremes` and `HybridPairedEIG` acquisition functions
    (`src/xac/acquisition_functions/acquisition_functions.py`);
  - refit restructuring in
    `src/xac/experimental_designs/experiment_runner.py`: methods fit only
    where selection needs a fit (ShaplEIG every iteration, hybrid at
    geometric adaptivity rounds after the extremes handover, fixed arms only
    at readout), while the per-budget metrics trace is produced by separate
    evaluation-only fits recorded in `eval_fit_duration`;
  - per-budget timing for the GP + Leverage baseline and export of its
    sampled design, so it appears in all three figure panels;
  - sweep configs `src/xac/experiments/conf/repro_all_*.yaml`.
- `gp_core.py`, `efficient_greedy.py`, `greedy_order.py` — standalone
  frozen-kernel greedy: dense reference, poly-time orbit version, closed-form
  validation.
- `gen_schedules.py` — writes `shapleig-repo/data/paired_schedule_p{9,14,16}.npy`
  (p=10 is committed).
- `repro_plots.py` — aggregates hydra multiruns into the per-game
  three-panel figures (MSE, method compute, selection size).
- `prep_games.py`, `benchmark.py`, `plot_results.py`, `smoke.py` — standalone
  benchmark on locally precomputed games in `games/`.
- `*.sbatch` — Slurm wrappers (schedules, smoke, full sweeps, aggregation).

## Data

`shapleig-repo/data/shapiq_games/` vendors the precomputed games from
[mmschlk/shapiq](https://github.com/mmschlk/shapiq) at commit `799cfd0f`
(the last commit before `data/precomputed_games` was removed upstream):
dataset-valuation games (Bike Sharing GB/RF, California Housing GB, p=10)
and local-explanation games (ViT 9 and 16 patches, ResNet-18 with 14
superpixels), 30 seeds each.

## Running

The experiment environment is the authors' (`poetry` spec in
`shapleig-repo/pyproject.toml`; a `uv`/pip venv with the same pins works).
From `shapleig-repo/` with `PYTHONPATH=src`:

```bash
python -m xac.experiments.cli --config-path conf --config-name repro_all_dv10
python repro_plots.py <multirun_root> [<multirun_root2> ...]
```

The sbatch files show the exact sweep shapes (games x 5 acquisitions x 30
seeds, 512 iterations).
