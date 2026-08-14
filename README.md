# shapleig-order

ShaplEIG [RFMBF, ICML '26] estimates Shapley values by fitting a Gaussian
process to observed coalition values and greedily evaluating the coalition
with the highest expected information gain about the Shapley vector. The
full writeup of this project is `shapleig_greedy_note.md`; in brief, it
establishes and then exploits the following fact.

**If the lengthscales are never refit, the "adaptive" order is
deterministic.** Gaussian conditioning changes the posterior covariance only
through *which* coalitions were evaluated, never through the observed
values, and the EIG reads only the posterior covariance. So with the kernel
frozen, the entire selection sequence is determined before the first
evaluation. The note computes this frozen-kernel sequence in closed form
(everything diagonalizes in the parity basis, and only odd parities are
visible to Shapley values): first the empty and grand coalitions, then
singleton and co-singleton pairs through all players, then complement pairs
of half-size coalitions with balanced intersections. Every step is followed
by its complement, and the extremes-first shape mirrors the Shapley kernel
weights that Kernel SHAP and Leverage SHAP [MW, ICLR '25] exploit. Logging
the adaptive rule's own selections shows it drifting along exactly this
schedule; the refits control the timing of the extremes-to-middles
migration, not the shape of the design.

**The hybrid approach.** Since the first $2p+2$ selections are justified
under any lengthscales, the hybrid arm replays the extremes from the fixed
schedule with no fitting at all, then hands over to EIG selection with
hyperparameter refits only at iterations $0, 1, 2, 4, 8, \dots$ past the
handover (about ten fits per 512 iterations instead of 512). Between refits
the EIG updates incrementally under frozen hyperparameters, which is the
paper's own scalability mode. The replication shows this keeps ShaplEIG's
accuracy at a fraction of its fitting cost, while the pure fixed schedule
(no adaptivity at all) matches it at most budgets.

Concretely, against ShaplEIG (`EIGFunctionProperty`), the hybrid arm
(`HybridPairedEIG`, which subclasses it) differs in exactly three places,
all in `shapleig-repo/src/xac`:

1. **When it fits the GP** (`experimental_designs/experiment_runner.py`).
   ShaplEIG refits every iteration in this project's own sweeps
   (`repro_all_*.yaml` leave `fit_config.refit_schedule` at its `MLMConfig`
   default of `None` and set `refit_interval: 1`). The GP-fitting code does
   support banded refit schedules (`refit_schedule: init_64_factor_4`, refit
   every iteration for the first 64 then every 8th/16th/32nd in successive
   bands; also `"geometric"`) — used by the upstream authors' own tree-model
   configs (`shapleig_crv_tree_*.yaml`, unrelated datasets and acquisition
   set, not part of this project's figures) — but none of the configs behind
   `figures/all_games.png` set it. The hybrid arm never fits during the
   extremes phase, then fits only at $0, 1, 2, 4, 8, \dots$ iterations past
   the handover to EIG selection — a schedule specific to the handover
   point, set directly in `experiment_runner.py`'s `HybridPairedEIG`
   branch, not via `refit_schedule`.
2. **What it selects while unfitted** (`acquisition_functions.py`).
   ShaplEIG always argmaxes EIG. The hybrid arm reads the next coalition off
   the fixed extremes schedule until none remain, then falls back to
   *exactly* ShaplEIG's own EIG argmax — `HybridPairedEIG.__call__` calls
   `super().__call__()` once extremes are exhausted.
3. **Whether `compute_AEA_new` uses `compute_AKZZA_fast`**
   (`applications.py`, gated by `application.use_akzza_fast`, see
   "AKZZA speedup" below). This isn't inherent to either acquisition rule —
   it's a config flag currently set only on the hybrid arm's sweep configs.
   `compute_AKZZA_fast` is a pure exact reformulation, validated
   independently of which acquisition function calls it, so nothing besides
   the config stops ShaplEIG from using it too.

A fourth difference follows mechanically from the first. Every iteration
needs a "metrics-trace readout": the Shapley-value estimate (and its MSE
against ground truth) a user stopping *right now* would actually get —
this is a different question from "which point should I evaluate next,"
which is all the acquisition rule's own (possibly stale, for efficiency)
internal GP needs to answer, so the two are allowed to use different
hyperparameters (`experiment_runner.py:426-431`). Since ShaplEIG always
just refit the GP it needs for selection, its readout can reuse that same
fit for free. The hybrid arm's readout, by contrast, needs its *own*
separate fit on almost every iteration (whenever it didn't just refit for
selection — i.e. during the extremes phase or between geometric-schedule
refits), recorded in `eval_fit_duration` rather than the method's own
`hp_fit_duration`, and added back on top of selection cost in the "compute
by evaluations" panel below (a real user does pay for it). This is exactly
the call site `compute_AKZZA_fast` helps most, and why the speedup shows up
far more in the hybrid arm's wall-clock profile than it would in ShaplEIG's
today.

**AKZZA speedup.** `compute_AKZZA_new` (the closed-form
$\mathbf{A}K_\xi(\mathbf{Z},\mathbf{Z})\mathbf{A}^\top$ computation, the
single largest cost in every arm's acquisition call) rebuilds a full
per-player forward pass from scratch for every player even though that pass
never depends on which player is excluded, and reruns its backward pass to
depths its own forward pass never uses. `compute_AKZZA_fast`
(`applications.py`) precomputes each pass once globally and reuses it per
player — a validated exact reformulation (not an approximation:
`torch.allclose` to float64 noise across $p=2..32$), 1.4-1.6x faster per
call, with zero accuracy cost. It is opt-in per application via
`use_akzza_fast: true` (default `False`, so every arm's previously
published numbers stay exactly reproducible unless a config opts in); only
the hybrid arm's sweep configs (`repro_all_*_hybrid_akzza.yaml`) currently
set it. See `applications.py`'s `compute_AKZZA_fast` docstring for the
derivation and `HANDOFF_AKZZA.md` for the full validation record.

## Figures

`figures/all_games.png` (with `.svg` alongside) is one mega-figure,
regenerated by `repro_plots.py` from the hydra sweeps: one row per game,
four columns.

1. **Error by evaluations**: MSE against the exact Shapley values
   (mean with SEM over 30 seeds) at every budget.
2. **Compute by evaluations**: method compute to reach each budget, meaning
   cumulative selection compute (the fits the method needs for selection,
   plus acquisition) plus the readout fit a user stopping at that budget
   would pay. For GP + Leverage the recorded per-budget duration is already
   that total, since the method restarts from scratch at every budget.
3. **Compute-error tradeoff**: the previous two traces against each other,
   one point per budget with the final budget marked, so equal-compute and
   equal-error comparisons can be read directly.
4. **Selected coalition by order**: the distance $\min(|T|, p-|T|)$ of each
   iteration's selected coalition from the nearest size extreme, showing
   where each rule samples over the course of the run.

Arms: ShaplEIG (refit every iteration), the hybrid above (single arm as of
2026-08-14 — see "Hybrid vs. ShaplEIG"; previously plotted as two variants,
exact-fit and `AcceleratedFitConfig` "fast fit", collapsed into one after
the fast-fit GP-fitting acceleration was found unreliable, see
`HANDOFF_AKZZA.md`), the fixed paired schedule (plain and `AcceleratedFitConfig`
"fast fit" — this one arm still runs both variants), GP + leverage score
sampling, and GP + random.

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
    where selection needs a fit (ShaplEIG every iteration, hybrid at the
    geometric adaptivity rounds after the extremes handover, fixed arms only
    at readout), while the per-budget metrics trace is produced by separate
    evaluation-only fits recorded in `eval_fit_duration`;
  - per-budget timing for the GP + Leverage baseline and export of its
    sampled design, so it appears in all three figure panels;
  - `compute_AKZZA_fast` and the `application.use_akzza_fast` gate
    (`src/xac/applications/applications.py`) — see "AKZZA speedup" above;
  - `AcceleratedFitConfig` and `fit_accelerated`
    (`src/xac/surrogates/fast_fit.py`) — a separate, independent
    acceleration of the GP *hyperparameter fit* itself (CG/Lanczos +
    inducing points + odd-part folding), only ever used by `PairedExtremes`
    (see "AKZZA speedup" above for why it was dropped from the hybrid arm);
  - sweep configs `src/xac/experiments/conf/repro_all_*.yaml`, including
    `repro_all_*_hybrid_akzza.yaml` (the current hybrid arm: exact fit +
    `use_akzza_fast: true`, `HybridPairedEIG` only) and `repro_all_*_fastfit.yaml`
    (the retired hybrid `AcceleratedFitConfig` variant, kept for
    `PairedExtremes+fast`, no longer used for `HybridPairedEIG`).
- `repro_plots.py` — aggregates hydra multiruns into `figures/`.
- `greedy/` — the frozen-kernel greedy itself: `gp_core.py` (dense
  reference implementation and self-tests), `efficient_greedy.py`
  (poly-time orbit version with closed-form validation), and
  `gen_schedules.py`, which writes
  `shapleig-repo/data/paired_schedule_p{9,14,16}.npy` (p=10 is committed).
- `slurm/` — Slurm wrappers (schedules, smoke, full sweeps, aggregation);
  job logs go to `run_logs/` (not tracked).

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
python ../repro_plots.py <multirun_root> [<multirun_root2> ...]
```

The `slurm/` scripts show the exact sweep shapes. Three config families by
game group (`repro_all_dv10`, `repro_all_vit9`, `repro_all_p1416`, the last
two covering the local-explanation games), each with three variants:

- `repro_all_*.yaml` — the base sweep: games x 5 acquisitions (ShaplEIG,
  Hybrid, PairedExtremes, GP+Leverage, GP+Random) x 30 seeds, exact MLM fit.
  Its `HybridPairedEIG` runs are no longer used in the figure (superseded by
  `_hybrid_akzza`, see below) but the sweep still generates them; only the
  other four arms' output feeds `figures/all_games.png`.
- `repro_all_*_fastfit.yaml` — games x 1 acquisition (`PairedExtremes` only
  as of 2026-08-14) x 30 seeds, `AcceleratedFitConfig`. Feeds the
  "Non-adaptive, fast fit" arm.
- `repro_all_*_hybrid_akzza.yaml` — games x 1 acquisition (`HybridPairedEIG`
  only) x 30 seeds, exact MLM fit, `application.use_akzza_fast: true`. Feeds
  the single "Hybrid" arm in the figure.

`repro_plots.py`'s `collect()` drops any `HybridPairedEIG` run (from either
of the first two families) that doesn't have `use_akzza_fast` set, by config
field rather than by which root it came from — so all roots (old and new)
can be passed to it together safely; see `sweep_roots_v2.txt` /
`sweep_roots_hybrid_akzza.txt` (both gitignored, list of multirun roots) and
`slurm/aggregate_v3.sbatch` for how the current figure is built.
