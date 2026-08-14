# Handoff: speed up `compute_AKZZA_new` (2026-08-13)

## Read these first

1. **The `hopper` skill.** Everything here runs via `sbatch`, never the login
   node — the venv, partitions (`debug` for cheap/short jobs, `main` for
   anything real), and the sbatch-script conventions this repo already uses
   are all covered there. Every script referenced below follows that
   pattern; copy one rather than inventing a new shape.
2. **The `how-to-write-and-think` skill.** Not for prose — for how to
   approach the math before touching code. The relevant moves: find the
   picture before the formula (Section 2 below gives one), look for the
   disguise (is the per-player recomputation actually a leave-one-out of
   something computable once, globally?), and name exactly what breaks
   before proposing a fix, not "this seems slow." A previous implementer
   already tried vectorizing this function and left a comment saying it
   didn't help — find out *why* before trying again, or you'll rediscover
   the same dead end.
3. **This whole file**, then `HANDOFF.md` in this same directory (the prior
   handoff — different task, same repo, same conventions, useful
   background on the fast-fit surrogate work this grew out of).
4. `shapleig_greedy_note.md` (repo root) — the math background for the
   ESP-based ("elementary symmetric polynomial") Shapley computation that
   `compute_AKZZA_new` implements. Section headers: closed form for the
   frozen-kernel greedy, how the greedy moves in distribution, making the
   lengthscales cheap. Read before touching the DP.

## The task, concretely

`compute_AKZZA_new` (`shapleig-repo/src/xac/applications/applications.py`,
currently ~line 1651) is the single biggest remaining cost in
`HybridPairedEIG`'s per-iteration acquisition-function call, and it is
**not sped up at all** by the existing `AcceleratedFitConfig` ("fast fit")
machinery — its cost depends only on the GP kernel's hyperparameters
(lengthscale, outputscale), never on archive size or which fitting
procedure produced those hyperparameters. Direct profiling on real
archives (p=10/14/16) measured **13-35ms per call**, growing with `p`,
called ~150+ times over a single 512-iteration Hybrid run (mostly from the
per-iteration readout path — see below), and this dominates
`acq_fun_duration`, itself the majority of Hybrid's total selection cost.

**Why this matters:** the original ask in this conversation was "make
Hybrid + fast fit faster than plain Hybrid." Two rounds of Cholesky-related
optimizations (incremental Cholesky update in `compute_AEA_new`, cached-
factor reuse in `compute_ASigmaW_new` — both already built, see "Current
state" below) targeted the archive-size-dependent part of the cost and
were correctly gated to `AcceleratedFitConfig` only, but both measured
**~1-2% real speedup** because they weren't the dominant cost.
`compute_AKZZA_new` is. Speeding it up will **not** by itself make fast
beat exact (its cost is identical for both — there's no `fit_config`
dependency to gate on), but it will make **every** Hybrid/ShaplEIG arm
faster in absolute terms, exact and fast alike, which has its own value
independent of that original framing. Confirm this reasoning still holds
before spending real effort — check `git log`/this file's own commit for
whether the framing changed after this was written.

## Current state (uncommitted)

```
$ git status --short
 M shapleig-repo/src/xac/applications/applications.py
 M shapleig-repo/src/xac/experimental_designs/experiment_runner.py
?? profile_acq_fun.py
?? profile_akzza.py
?? validate_incremental_cholesky.py
?? validate_incremental_cholesky_mse.py
?? slurm/profile_acq_fun.sbatch
?? slurm/profile_akzza.sbatch
?? slurm/validate_incremental_cholesky.sbatch
?? slurm/validate_incremental_cholesky_mse.sbatch
```

The two modified files contain the validated-safe (but modest) Cholesky
work: `_cholesky_maybe_incremental` and `_solve_against_cached_cholesky` in
`applications.py` (both gated to `AcceleratedFitConfig`, both confirmed
mathematically exact per-step, both confirmed aggregate-MSE-safe via a
real 8-seed/512-iteration resnet_14 comparison), plus `_L_cache`/
`_L_cache_chain_len` added to `experiment_runner.py`'s `_EIG_CACHE_ATTRS`
protection list. Decide whether to commit these alongside the
`compute_AKZZA_new` work or separately — they're independent and both
already validated, so there's no reason to block one on the other.

The `profile_*.py` / `validate_*.py` scripts (and their `slurm/*.sbatch`
counterparts, all following the same copy-and-`sed` pattern visible in
`slurm/` — `cd` into `shapleig-repo/`, invoke the sibling `../foo.py`) are
disposable diagnostics, kept for reference and reuse:

- `profile_akzza.py` — cProfile breakdown of `compute_AKZZA_new` in
  isolation (builds a throwaway `GPSurrogate` + a bare
  `ShapiqShapleyApplication.__new__(...)`, no real blackbox needed). This
  is where the 81%-self-time, no-dominant-sub-op finding came from. Start
  here to re-confirm the current cost profile before changing anything —
  re-run it after any change to see the delta directly, without needing a
  full pipeline run.
- `profile_acq_fun.py` — timing wraps around
  `EIGFunctionProperty.__call__`'s real sub-calls on an actual smoke-scale
  HybridPairedEIG run. This is what showed `compute_AKZZA_new` eating 91%
  of `compute_AEA_new`'s time and being called more often than
  `EIGFunctionProperty.__call__` itself (i.e., mostly from the readout
  path, not the main acquisition path — see "Where the calls come from"
  below).
- `validate_incremental_cholesky.py` — end-to-end correctness harness:
  runs the same seed twice (a toggle forces one path vs. the other) via
  `hydra.compose` + direct `run_experiment(...)` calls (bypassing
  `cli.py`'s `@hydra.main` decorator). **Read the docstring at the top —
  it documents a real methodological trap**: bit-identical archive
  trajectories turned out to be the wrong correctness bar for anything
  touching `HybridPairedEIG`'s greedy selection (see "Validation
  gotchas" below). Reuse this pattern, but reuse the *lesson*, not
  necessarily the bit-identical assertion.
- `validate_incremental_cholesky_mse.py` — the actual bar that matters:
  N-seed parallel (via `joblib.Parallel`) real-scale comparison, reporting
  mean±SEM final MSE and total selection time per condition. This is what
  should gate whether any `compute_AKZZA_new` change is safe to land.

## The picture: what `compute_AKZZA_new` actually computes

Concretely, before the notation: for a fixed set of per-player kernel
parameters `(alpha_i, beta_i)` (derived from the fitted lengthscales —
see `_get_kernel_alpha_beta_new`), `AKZZA[i, j]` is a closed-form
aggregate over **all coalitions** of a bilinear form involving players `i`
and `j`, computed via the "even-odd" / ESP decomposition documented in
`shapleig_greedy_note.md` rather than by ever enumerating `2^p` coalitions
directly. The diagonal (`i == j`) and off-diagonal (`i != j`) entries are
each built from a **leave-one-or-two-players-out** product over the
remaining `p-1` (or `p-2`) players' `(alpha, beta)` pairs — that's the
"disguise" to look for: is this actually one global product with two
players removed, computable from a single forward+backward pass over all
`p` players once, rather than `p` independent backward passes (one per
excluded player `i`) as the current code does?

## Where the calls come from (this changes the fix)

`compute_AEA_new`'s own no-refit-step cache
(`if is_no_refit_step and hasattr(self, "AKA")`) already skips
`compute_AKZZA_new` on the **main acquisition path** when hyperparameters
haven't changed. But `profile_acq_fun.py`'s call counts
(`compute_AKZZA_new` called 151 times against only 124
`EIGFunctionProperty.__call__` invocations) show most calls are **not**
hitting that cache — almost certainly from the readout path:
`_readout_property_posterior` → `application.property_posterior(...,
recycle=False)` → `compute_AF_PPV` → `compute_AEA_new(surrogate,
scale_by_emp_std=True, precomputed_A_KZX=...)` with **`is_no_refit_step`
left at its default of `False`**, forcing the `else` branch every single
readout call. And the readout surrogate is cold-started fresh every
iteration (a design choice flagged, but not touched, earlier in this
conversation — see the main session transcript around where
`eval_fit_duration`/readout cost was first identified), so its
hyperparameters genuinely differ call to call, meaning the *existing*
`AKA`-reuse cache legitimately can't help there even if it were checked.

Two independent angles worth separating:

1. **Make `compute_AKZZA_new` itself faster** (the O(p²)→O(p) algebraic
   idea above, or reducing per-call Python/tensor-dispatch overhead —
   both discussed below). Benefits every call, everywhere.
2. **Question whether the readout path needs a *fresh* `AKZZA` at all.**
   `AKZZA` depends only on `(alpha, beta)`, which depend only on
   lengthscale/outputscale. If two consecutive readout fits happen to
   converge to very similar hyperparameters (plausible for a warm,
   slowly-changing archive), a hyperparameter-keyed cache (round
   lengthscale/outputscale to some tolerance, cache-hit if unchanged since
   last readout) could skip recomputation entirely on many iterations.
   This is a different, possibly higher-leverage lever than the DP
   rewrite, and lower-risk (a cache with a tolerance-based key, falling
   back to full recompute on any miss, is a much smaller correctness
   surface than rewriting the recurrence). Worth profiling first: log how
   often consecutive readout hyperparameters actually land within, say,
   1e-3 relative of each other before investing in the DP rewrite.

## Evidence: what's slow and what isn't

From `profile_akzza.py` (p=14, 20 reps, cProfile sorted by self-time):

```
compute_AKZZA_new (self time only):  22.95ms/call  (81% of 28.2ms total)
torch.einsum (all calls combined):    2.7ms/call
abs/clamp_min/max/log (combined):     ~2ms/call
.item() (all calls combined):        ~0.2ms/call
```

No single named sub-operation dominates — the cost is spread across many
small tensor ops (`a_j * S_right_prev`, slice-assign `+=`, `torch.zeros`,
indexing) inside the nested
`for i in range(p): for k in range(nf-1,-1,-1): ...` /
`for m in range(nf): ...` loops, none of which get their own cProfile
frame (PyTorch's C++ dispatch for basic tensor arithmetic and slicing
doesn't create a separate Python call frame the way `torch.einsum`'s
Python wrapper does — this is *why* the cost shows up as
`compute_AKZZA_new`'s own "self time" rather than under some named
culprit). `line_profiler` is not installed in `.venv-repro`; installing it
(`uv pip install line_profiler` inside the venv, or add to whatever the
env's lockfile is) would give a true line-by-line breakdown and is
probably the fastest way to confirm exactly which lines cost the most
before attempting a rewrite — worth doing before guessing further.

For `p=10/14/16`: 13.2ms / 26.3ms / 35.5ms per call respectively — clearly
growing with `p`, consistent with the nested-loop structure being
`O(p^2)` bodies each doing `O(p)`-sized tensor work (tables are
`(p+1, p+1, 4)`).

## Two candidate approaches (not mutually exclusive)

**A. Algorithmic: O(p²) → O(p) via a global leave-one-out pass.** The
current code reruns a full backward pass over `p-1` factors independently
for *each* of the `p` values of `i`. If the "picture" above is right (each
`i`'s computation is a leave-one-out product), a single global forward
pass and a single global backward pass over all `p` players, combined
pairwise (`prefix[i-1] ⊗ suffix[i+1]`) for each `i`, would suffice — this
is the standard trick for "product/aggregate of all except one" problems.
**This needs to be derived and checked on paper (or in a scratch
notebook) before writing any code** — the composition operator here isn't
plain multiplication (it's built from `(alpha, beta)` pairs folded
through 4-wide coefficient tables with numerical-stability rescaling), so
confirm it's actually associative in the needed sense before assuming
the standard trick applies unmodified. This is exactly the "find the
disguise, then name what breaks" move from `how-to-write-and-think` — do
that reasoning explicitly, in writing, before touching
`applications.py`.

**B. Mechanical: reduce per-call dispatch overhead without changing the
algorithm.** Lower-risk, likely smaller payoff. Ideas, roughly in order of
expected safety:
- Install `line_profiler` and get the real per-line breakdown first,
  rather than guessing.
- `log_S`/`log_P`/`scale` are currently tracked as **Python floats**,
  requiring a `.item()` (tensor → Python scalar) on every loop iteration
  and a `math.exp()` (Python-level) instead of `torch.exp()`. Keeping
  them as 0-d tensors throughout and only touching Python at the very end
  (or not at all — `AKZZA[i, j] = some_0d_tensor` works fine in PyTorch)
  removes some overhead, but per the profiling `.item()` itself is only
  ~0.2ms/call combined — this alone won't explain the 20ms+ "self time,"
  so don't expect much from this in isolation.
- `torch.compile` on the per-`i` inner loop body: PyTorch 2.9 (confirmed
  installed) supports this, and it's specifically aimed at exactly this
  failure mode (many small CPU ops, Python-dispatch-bound). The
  obstruction: `.item()` calls force "graph breaks" that limit fusion —
  approach B's `.item()`-removal may be a **prerequisite** for `compile`
  to help rather than a standalone win. Also worth checking: does
  `torch.compile`'s own compilation overhead (paid once per distinct `p`,
  per process) net out favorably inside a 90-worker `joblib` sweep where
  each worker is a fresh process? Measure before committing to this path.
- A prior attempt at vectorizing across `i` was tried and rejected
  ("not faster due to memory overhead," per the existing code comment).
  Understand *why* before retrying — was it naive broadcasting to a
  `(p, p+1, p+1, 4)` tensor (176k elements for p=14, not obviously
  memory-prohibitive), or something else? If you can't find out why,
  re-attempt small and measure rather than trusting the comment blindly
  — codebases accumulate comments about failed experiments whose exact
  conditions get lost.

## Validation protocol (do not skip steps — each one caught a real issue)

This function is used by **every** arm (ShaplEIG, exact Hybrid, fast
Hybrid) — unlike the Cholesky work, there's no `AcceleratedFitConfig` gate
protecting the exact baseline from a bug here. Hold this to a stricter bar
than "aggregate MSE looks fine": since this rewrite should be an *exact*
reformulation (not a numerical approximation like the incremental
Cholesky), **first** confirm near-bit-identical numerical agreement
against the original implementation directly, on a battery of synthetic
`(p, alpha, beta)` triples spanning small (`p=3-5`, hand-checkable) through
production (`p=9,10,14,16`) scale, before ever running it inside a real
experiment. A `torch.allclose` battery across many random draws is cheap
and should be the first gate, not the multi-seed MSE comparison.

Only after that passes, repeat the same two-stage loop already established
in this session:

1. **Cheap correctness smoke test** (`validate_incremental_cholesky.py`
   pattern) — confirms no crashes/shape errors on a real HybridPairedEIG
   run. Fast (~10 min), always run first.
2. **Real multi-seed MSE + timing comparison**
   (`validate_incremental_cholesky_mse.py` pattern) — the bar that
   actually matters for whether this is safe to land. Expensive (~1hr per
   run at production scale in this session's experience) — budget for it,
   don't skip it, and don't over-interpret a single seed's result (only
   the aggregate mean±SEM comparison is meaningful, per the lesson
   below).

### Specific gotchas hit while building the Cholesky work (all real, all cost real time to find)

- **Bit-identical archive trajectories is the wrong correctness bar.**
  `HybridPairedEIG`'s greedy selection is sensitive enough that *any*
  nonzero floating-point difference — however tiny, however
  mathematically justified — will eventually land on a near-tied EIG
  comparison and flip which point gets selected, and once one iteration
  diverges, every iteration after it does too (confirmed: the *exact
  same* forced-full computation run twice was bit-identical, ruling out
  hidden non-determinism — the divergence is real and caused by whatever
  changed, but it is not evidence of a bug). Judge correctness by
  aggregate MSE across seeds, matching how the actual published figure
  itself validates arms against each other — not by whether the archive
  matches.
- **Must call `set_seed(cfg.meta.seed)` explicitly** when bypassing
  `cli.py`'s `@hydra.main`-decorated `main()` (which normally does this
  first thing) — a validation script that does its own `hydra.compose` +
  direct `run_experiment(...)` calls will silently inherit whatever
  random state a *previous* call in the same process left behind
  otherwise, corrupting exactly this kind of two-condition comparison
  (this alone produced a spurious 249/461-selections-differ result before
  being caught).
- **`joblib.Parallel`'s default (loky) backend reuses worker processes
  across tasks.** Any monkeypatch applied conditionally inside a
  `delayed(...)` function must be restored in a `finally` block
  regardless of which branch ran, or it leaks into a later, unrelated
  task that happens to land on the same reused worker.
- **A chain/iteration cap tuned at the wrong scale can silently throttle
  away most of the benefit.** (Specific to the Cholesky work, but the
  general lesson applies: any safety cap should be validated at
  *production* scale, not just smoke scale — a cap that looks
  conservative-and-safe at `t~250` fully negated the speedup at `t~500`
  because it forced resets far more often than necessary there.)
- **Any new per-application cross-call cache must be added to
  `_EIG_CACHE_ATTRS`** in `experiment_runner.py` — the readout path uses
  a *different* surrogate (different hyperparameters, same archive size)
  and snapshots/restores that attribute list around itself specifically
  so its own cache-population doesn't leak into the main acquisition
  path's next call. If `compute_AKZZA_new`'s rewrite adds new cached
  state (e.g., a hyperparameter-keyed AKZZA cache per the readout-caching
  idea above), it needs the same protection or it will silently produce
  wrong results the next no-refit iteration on the main path.

## Suggested plan of attack, with subagent delegation

This is exploratory + correctness-critical + math-heavy — the kind of
task where a fresh, focused context for the derivation step pays off, and
where the correctness-battery testing is embarrassingly parallel. Suggested
structure (adapt freely; this is a starting point, not a script to follow
blindly):

1. **Solo, no subagent:** read the four "read first" items above in full.
   Re-run `profile_akzza.py` to confirm current numbers haven't drifted
   (git state, torch version, etc. could have changed).
2. **Dispatch a research/derivation subagent** (general-purpose,
   foreground — you need its answer before proceeding, and it benefits
   from a completely fresh context uncluttered by the rest of this
   session's history) with a self-contained prompt: give it
   `compute_AKZZA_new`'s exact current source (line range), the
   `shapleig_greedy_note.md` background, and ask it to (a) restate in
   plain language what `AKZZA[i,j]` computes (the "picture"), (b)
   determine whether the leave-one-out structure decomposes into a single
   global forward+backward pass (the "disguise"), and (c) if yes, write
   out the exact recurrence for the rewrite on paper/in markdown — no
   code yet. Do not let it write code in this pass; the point is getting
   the math right in isolation, checked by a human (you) before it
   touches anything.
3. **Solo or a coding subagent:** implement the derived rewrite (or, if
   the derivation in step 2 says the algorithmic approach doesn't hold up,
   fall back to approach B's mechanical reductions) as a **new, separate
   function** alongside the original — do not replace in place yet.
4. **Correctness battery** (this is the embarrassingly-parallel part —
   consider `Workflow` or parallel `Agent` calls if you want breadth
   quickly, e.g., one task per `(p, seed)` combination checking
   `torch.allclose` between old and new): many `(p, random alpha/beta)`
   draws, `p` spanning 3 through 16, comparing old vs. new numerically.
   Every mismatch beyond float64 tolerance is a stop-and-fix, not a
   log-and-continue.
5. **Swap in, rerun the full validation protocol** above (cheap smoke
   test, then real multi-seed MSE+timing comparison) before considering
   this done.
6. **If validated:** this is a pure speed change to *already-correct*
   math, so existing committed baseline/fast-fit data remains scientifically
   valid as-is — there's no need to regenerate published figures for
   correctness, only if you want the (identical) numbers to have been
   produced faster. Commit with a clear message explaining the
   before/after profiling numbers (matching the style of the fast-fit
   commits already in this repo's history — `git log` for examples), push,
   and send the same kind of completion notification this session has
   been sending throughout (the user has explicitly asked for these on
   this project).
