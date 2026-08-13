# Handoff: fast-fit ShaplEIG variants (2026-08-12)

## Task

Teal asked for three ShaplEIG method variants to be run and compared:

1. **ShaplEIG as implemented by the authors** — unchanged (`EIGFunctionProperty`,
   refit every iteration, exact MLM fit). Reusing existing baseline data
   (see "Baseline data" below), not re-run.
2. **Non-adaptive, single fast fit** — `PairedExtremes` acquisition (never
   needs a fit for selection) paired with the new `AcceleratedFitConfig`
   surrogate.
3. **Hybrid + fast fit** — `HybridPairedEIG` (fixed extremes, then
   geometric-schedule EIG) paired with the same `AcceleratedFitConfig`.

Requested speedup techniques for the fit: (2) iterative CG/Lanczos, (3)
sparse inducing points, (5) odd-part folding under complement pairing —
numbering matches the earlier conversation about GP-fitting complexity.

When the full reruns land: **regenerate `figures/all_games.{png,svg}`,
commit everything, push, and send Teal a push notification** (they
explicitly asked for the notification).

## Status as of this handoff

- **267864** (dv10 fastfit): **COMPLETE**, verified clean (180/180 unique
  (game, acq, seed) combos, all pure dv10 games, no duplicates).
- **267758** (vit9 fastfit): **COMPLETE**, verified clean (60/60, pure
  vit_9, no duplicates).
- **267759** (p1416 fastfit, resnet_14 + vit_16): **STILL RUNNING** as of
  this handoff (started ~13:29, 24h time limit, `main` partition). Check
  with:
  ```
  sacct -j 267759 --format=JobID,State,Elapsed,Timelimit
  ```
  If a background watcher notification isn't available (fresh session),
  poll `sacct` directly, or re-arm a watcher:
  ```
  until sacct -j 267759 --format=State --noheader | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_ME'; do sleep 300; done
  ```

## What's implemented (code, all under `shapleig-repo/src/xac/`)

- **`surrogates/fast_fit.py`** (new file) — `AcceleratedFitConfig` (an
  `MLMConfig` subclass) and `fit_accelerated()`. Fits an *auxiliary* model
  fast, then copies (lengthscale, outputscale) onto the real exact model —
  the downstream ESP-based Shapley math (`applications.py`) never changes.
  Read its module docstring; it documents the three techniques and two
  important corrections made after empirical testing (see "Bugs found and
  fixed" below).
- **`surrogates/gp_surrogate.py`** (edited) — `GPSurrogate.fit()` gets one
  new branch: if `self.config.fit_config` is an `AcceleratedFitConfig`,
  call `fit_accelerated()`; on any exception, fall through to the ordinary
  exact fit (never crashes a sweep).
- **`surrogates/__init__.py`** (edited) — exports `AcceleratedFitConfig`,
  `OddKernel`.
- **`repro_plots.py`** (edited, at `shapleig-order/` root, *not* inside
  `shapleig-repo/`) — `collect()` now appends `"+fast"` to the acquisition
  key when `surrogate.fit_config._target_` ends in `AcceleratedFitConfig`,
  so exact vs. fast-fit arms plot as distinct lines. Two new `ARMS`
  entries added with colors `#059669` (non-adaptive fast) and `#4338ca`
  (hybrid fast), matching the existing muted palette.

## New configs (`shapleig-repo/src/xac/experiments/conf/`)

- `repro_all_dv10_fastfit.yaml`, `repro_all_vit9_fastfit.yaml`,
  `repro_all_p1416_fastfit.yaml` — clones of the corresponding
  `repro_all_*.yaml`, restricted to `acquisition._target_:
  PairedExtremes,HybridPairedEIG` and using
  `surrogate.fit_config._target_: xac.surrogates.AcceleratedFitConfig`
  with `inducing_points: null` (see below for why).
- `repro_all_smoke_dv10_fastfit.yaml`, `repro_scale_dv10_exact.yaml`,
  `repro_scale_dv10_fastfit.yaml` — smaller configs used only for
  validation; safe to delete before the final commit if you want a
  cleaner diff, or keep them (they're cheap and document the validation).

## New sbatch scripts (`slurm/`)

- `full_dv10_v2_fastfit.sbatch` (mem=500gb, see below), `full_vit9_v2_fastfit.sbatch`
  (mem=150gb), `full_p1416_v2_fastfit.sbatch` (mem=480gb) — launch the
  three full reruns. All three now snapshot `multirun/` before/after their
  `$PY -m xac.experiments.cli ...` call and diff to find their own output
  directory, instead of the original template's `ls -td multirun/*/* |
  head -1` (see "Bugs found and fixed").
- `smoke_fastfit.sbatch`, `smoke_fastfit_inducing.sbatch`,
  `smoke_fastfit_scale.sbatch`, `diag_memory.sbatch` (+ `diag_memory.py`
  at repo root) — validation/diagnostic scripts, not part of the
  deliverable. Safe to delete before commit, or keep for documentation.

## Bugs found and fixed during validation (all fixed in the current code)

1. **Inducing points are slower, not faster, at this problem's actual
   scale.** Measured directly: `InducingPointKernel`'s own
   construction/evaluation overhead exceeded a direct factorization at
   archive sizes up to ~500 — the O(t³) cost was never the bottleneck
   here. Fixed by defaulting `inducing_points: None` (disabled) in
   `AcceleratedFitConfig` and all fastfit configs. The code path is kept,
   documented, for a scale where that trade would flip.

2. **Forcing CG/Lanczos on a large, *unfolded* archive OOM'd a 90-worker
   job.** `HybridPairedEIG` well past its extremes handover has a mostly
   unpaired (EIG-selected) archive, so odd-part folding correctly
   self-disables there — but I was still forcing iterative solves on the
   full ~500-point unfolded system, and that combination is memory-hungry.
   Fixed: `force_iterative` now requires `used_fold=True`. Unfolded
   archives fall back to ordinary direct Cholesky (same cost the exact
   baseline already pays successfully at that size — no acceleration for
   that call, but no regression).

3. **That fix alone didn't stop the OOM** (same failure, same ~22min
   mark, on the *next* dv10 resubmit). A single-process, single-worker
   diagnostic (`diag_memory.py`) showed `fit_accelerated`'s own memory use
   is bounded (grew from ~340MB to ~450MB over 210 sequential fits, *not*
   a leak). Conclusion: the OOM is aggregate peak across 90 concurrent
   joblib workers — each accelerated fit builds a second (auxiliary) model
   on top of the real one, raising per-call peak memory versus the
   baseline's single-model fit, and all workers' archives grow in lockstep
   with iteration index so their peaks land in near-synchrony late in the
   run. Fixed by bumping `full_dv10_v2_fastfit.sbatch`'s `--mem` from
   200gb to 500gb (nodes have up to 3TB available, confirmed via
   `sinfo`). This resubmit (267864) completed cleanly.

4. **`ls -td multirun/*/* | head -1` races across concurrently-running
   jobs sharing the same `cd shapleig-repo` working directory** — it picks
   the most-recently-*created* directory across the *whole* shared tree,
   so a job whose own hydra run already finished (or crashed) can grab a
   *different*, concurrently-running job's fresher directory instead of
   its own. **Confirmed this actually happened**: the first dv10 OOM
   crash (267675) recorded a directory that, on inspection, held a mix of
   dv10 *and* vit_16 data from a job that started around the same time.

   First fix attempt (snapshotting `multirun/` before/after and diffing
   with `comm -13`) turned out **not** to be robust: it still infers the
   right directory from what's merely *new*, so two jobs launched inside
   the same wall-clock second can both see each other's directory as
   "new" and pick the wrong one. **Confirmed this happened too**: the
   267757 dv10 resubmit's own log recorded `15-19-14`, which was actually
   vit_9's directory (267758) — harmless that time only because vit_9's
   own script happened to log the same, correct path independently.

   **Real fix (now in all six `full_*.sbatch` scripts, fastfit and
   original)**: pass `hydra.sweep.dir=multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}-$SLURM_JOB_ID`
   so each job's directory name is unique by construction — no inference,
   no diffing, no race regardless of how many jobs start in the same
   second. Root detection afterward is just `find multirun -mindepth 2
   -maxdepth 2 -type d -name "*-${SLURM_JOB_ID}"`.

   Because of bug 4, `sweep_roots_v2.txt` was manually cleaned: removed
   the two lines pointing at the corrupted `multirun/2026-08-12/13-15-09`
   (the first-round dv10 OOM crash's misattributed root). Current clean
   content (5 lines): the 3 original baseline roots + the verified-clean
   vit9 fastfit root (`15-19-14`) + the verified-clean dv10 fastfit root
   (`17-16-54`). **Before appending p1416's root once it completes,
   verify it the same way** (don't trust the "root recorded" log line
   blindly — cross-check `blackbox.name`/`acquisition._target_` counts
   via each run's `.hydra/config.yaml`, expect exactly 120 configs: 60
   resnet_14 + 60 vit_16, 60 PairedExtremes + 60 HybridPairedEIG, no
   duplicate (game, acq, seed) triples). Verification snippet used
   throughout (adapt the glob path):
   ```python
   import glob, yaml
   from collections import Counter
   games, acqs, combos = Counter(), Counter(), Counter()
   for cfg_path in glob.glob("multirun/<date>/<time>/**/.hydra/config.yaml", recursive=True):
       cfg = yaml.safe_load(open(cfg_path))
       g, a, s = cfg["blackbox"]["name"], cfg["acquisition"]["_target_"], cfg["meta"]["seed"]
       games[g] += 1; acqs[a] += 1; combos[(g, a, s)] += 1
   print(games, acqs)
   print("dups:", {k: v for k, v in combos.items() if v > 1})
   ```

## Validated performance (from smaller-scale checks, before the bug fixes
   above but the fitting math itself is unchanged)

- At t≈300 (folded, no inducing points): **~2x speedup** (114.6s → 56.6s
  cumulative fit time) with equivalent-or-better MSE (4.0e-7 vs 4.8e-7).
- At t≤59: consistent ~2x speedup, zero optimizer failures, zero crashes.
- Paired sampling for the initial design (`pairing_trick=True`, hardcoded
  in `applications.py:694`, `ShapleyApplication.random_init_design`
  defaults `False`) is identical across all three variants — confirmed by
  inspection, not just assumption, since none of the fastfit configs touch
  `application`/`acquisition_optimizer` blocks.
- All jobs are CPU-only (verified: no `--gres=gpu` in any sbatch script,
  no `.cuda()` in the code, `.venv-repro`'s torch build is
  `2.9.1+cpu`).

## Remaining steps (in order)

1. Wait for 267759 (p1416) to reach a terminal state.
2. **Verify its result is clean** (see verification snippet above) before
   trusting it. If `ls -td`'s race condition bit it too (unlikely since
   nothing else should be running concurrently with it right now, but
   check), the fallback is to resubmit `full_p1416_v2_fastfit.sbatch`
   (now fixed) alone.
3. Append the verified-clean p1416 root to `sweep_roots_v2.txt`.
4. Run `sbatch slurm/aggregate_v2.sbatch` to regenerate
   `figures/all_games.{png,svg}` (reads all roots in
   `sweep_roots_v2.txt`, `main` partition, ~1h limit, cheap).
5. Sanity-check the regenerated figure (7 arms now: the original 5 plus
   "Non-adaptive, fast fit (CG+ind.pts+odd)" and "Hybrid + fast fit
   (CG+ind.pts+odd)").
6. Decide whether to delete the validation-only scratch files (`diag_memory.py`,
   `slurm/diag_memory.sbatch`, `slurm/smoke_fastfit*.sbatch`,
   `shapleig-repo/src/xac/experiments/conf/repro_all_smoke_dv10_fastfit.yaml`,
   `repro_scale_dv10_*.yaml`) before committing, or keep them — they're
   harmless either way, just not part of the core deliverable.
7. `git add`/`commit`/`push` from `shapleig-order/` (the repo root; note
   `shapleig-repo/` is *not* its own git repo — no `shapleig-repo/.git` —
   it's plain files tracked by the outer repo, confirmed). Current dirty
   state (`git status --short`) at time of writing:
   ```
    M repro_plots.py
    M shapleig-repo/src/xac/surrogates/__init__.py
    M shapleig-repo/src/xac/surrogates/gp_surrogate.py
   ?? diag_memory.py
   ?? shapleig-repo/src/xac/experiments/conf/repro_all_{dv10,p1416,vit9,smoke_dv10}_fastfit.yaml
   ?? shapleig-repo/src/xac/experiments/conf/repro_scale_dv10_{exact,fastfit}.yaml
   ?? shapleig-repo/src/xac/surrogates/fast_fit.py
   ?? slurm/{full_dv10,full_vit9,full_p1416}_v2_fastfit.sbatch
   ?? slurm/{diag_memory,smoke_fastfit,smoke_fastfit_inducing,smoke_fastfit_scale}.sbatch
   ```
   Plus, after step 4, `figures/all_games.png`/`.svg` will be modified
   (already tracked, not gitignored). `sweep_roots_v2.txt`, `run_logs/`,
   and `shapleig-repo/multirun/` are gitignored — don't force-add them.
8. **Send Teal a push notification** once committed and pushed — they
   explicitly asked for this (`PushNotification` tool, `status: "proactive"`).
   Keep it under 200 chars, e.g.: "shapleig fast-fit: all 3 jobs done,
   figures regenerated, pushed to main."

## Key context worth knowing before touching anything

- The `pairing_trick`/paired-sampling check earlier in this conversation
  was already confirmed correct by inspection — no code changes were
  needed for that.
- `shapleig-repo` is the authors' vendored repo (`slds-lmu/shapleig`,
  MIT, upstream commit `162ce44`, diff in `upstream-changes.patch`,
  original `.git-upstream` preserved locally but not pushed) — it's fine
  to keep editing files inside it; that's the established pattern here
  (see `README.md`'s "Main changes" list, which documents prior local
  modifications the same way).
- Everything runs via `sbatch`, never the login node, per the `hopper`
  skill. All jobs so far are CPU-only on `main`/`debug`.
