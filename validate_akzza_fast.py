"""Correctness + timing battery for ShapleyApplication.compute_AKZZA_fast
against the reference compute_AKZZA_new, per HANDOFF_AKZZA.md's validation
protocol: a torch.allclose battery across many (p, random alpha/beta)
draws is the first gate, before this new implementation is ever run inside
a real experiment. Random alpha/beta come from fitting real GP surrogates
on random archives at each p (same construction as profile_akzza.py), so
each (p, seed) draw exercises realistic hyperparameter ranges rather than
hand-picked values.
"""

import sys
import time

import torch

sys.path.insert(0, "src")

from xac.applications.applications import ShapiqShapleyApplication
from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig)

torch.set_default_dtype(torch.float64)


def build_application_and_surrogate(p, t, seed):
    torch.manual_seed(seed)
    cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=1e-6),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
        fit_config=AcceleratedFitConfig(),
    )
    baseline_config = torch.zeros(p, dtype=torch.float64)
    candidate_config = torch.ones(p, dtype=torch.float64)
    X = torch.randint(0, 2, (t, p), dtype=torch.float64)
    Y = torch.randn(t, 1, dtype=torch.float64)
    gp = GPSurrogate(
        X, Y, config=cfg, cat_dims=[], log_trafo_dims=[], bounds=None,
        shapley_configs=(baseline_config, candidate_config),
    )
    gp.fit()

    app = ShapiqShapleyApplication.__new__(ShapiqShapleyApplication)
    object.__setattr__(app, "amount_players", p)
    return app, gp


def main():
    p_values = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    seeds = [0, 1, 2, 3]

    worst_abs = 0.0
    worst_rel = 0.0
    n_checked = 0
    failures = []

    for p in p_values:
        t = max(20, min(4 * p, 200))
        for seed in seeds:
            app, gp = build_application_and_surrogate(p, t, seed)

            AKZZA_ref = app.compute_AKZZA_new(gp)
            AKZZA_fast = app.compute_AKZZA_fast(gp)

            if not torch.allclose(AKZZA_ref, AKZZA_fast, rtol=1e-8, atol=1e-10):
                diff = (AKZZA_ref - AKZZA_fast).abs()
                rel = diff / AKZZA_ref.abs().clamp_min(1e-300)
                failures.append(
                    (p, seed, diff.max().item(), rel.max().item())
                )

            diff = (AKZZA_ref - AKZZA_fast).abs()
            rel = diff / AKZZA_ref.abs().clamp_min(1e-300)
            worst_abs = max(worst_abs, diff.max().item())
            worst_rel = max(worst_rel, rel.max().item())
            n_checked += 1

        print(f"p={p:3d}: checked {len(seeds)} seeds, "
              f"running worst abs={worst_abs:.3e} rel={worst_rel:.3e}",
              flush=True)

    print(f"\n=== Correctness battery: {n_checked} (p, seed) draws checked ===")
    print(f"worst abs diff: {worst_abs:.3e}")
    print(f"worst rel diff: {worst_rel:.3e}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for p, seed, a, r in failures:
            print(f"  p={p} seed={seed}: abs={a:.3e} rel={r:.3e}")
    else:
        print("\nALL PASSED (torch.allclose, rtol=1e-8, atol=1e-10)")

    # Timing comparison at a few representative p (mirrors profile_akzza.py)
    print("\n=== Timing: compute_AKZZA_new vs compute_AKZZA_fast ===")
    for p in [10, 14, 16, 24, 32]:
        app, gp = build_application_and_surrogate(p, t=50, seed=0)

        _ = app.compute_AKZZA_new(gp)
        _ = app.compute_AKZZA_fast(gp)

        reps = 20
        start = time.perf_counter()
        for _ in range(reps):
            _ = app.compute_AKZZA_new(gp)
        t_old = (time.perf_counter() - start) / reps

        start = time.perf_counter()
        for _ in range(reps):
            _ = app.compute_AKZZA_fast(gp)
        t_new = (time.perf_counter() - start) / reps

        print(f"p={p:3d}: old={t_old*1000:7.2f}ms  new={t_new*1000:7.2f}ms  "
              f"speedup={t_old/t_new:.2f}x", flush=True)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
