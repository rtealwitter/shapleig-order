"""Diagnostic: why is AcceleratedFitConfig's readout fit SLOWER than the
exact fit on resnet_14/vit_16 (p=14/16) despite being faster on vit_9/dv10
(p=9/10)? Loads a real archive (t~500+) from an actual p1416 run and times
the exact fit vs. the accelerated fit under its current config (folded +
forced iterative) vs. a folded-but-direct-Cholesky variant, to check
whether forcing CG/Lanczos on top of odd-part folding is itself the
regression (mirrors the already-established "inducing points + iterative
don't stack" finding, for folding + iterative instead).
"""

import dataclasses
import json
import sys
import time

import torch

sys.path.insert(0, "shapleig-repo/src")

from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig, MLMConfig)

torch.set_default_dtype(torch.float64)


def load_archive(run_dir, n=None):
    with open(f"{run_dir}/metrics.json") as f:
        d = json.load(f)
    X = torch.tensor(d["archive_x"], dtype=torch.float64)
    Y = torch.tensor(d["archive_y"], dtype=torch.float64).reshape(-1, 1)
    if n is not None:
        X, Y = X[:n], Y[:n]
    return X, Y, d["blackbox"]


def time_fit(X, Y, fit_config, label, repeats=3):
    p = X.shape[1]
    cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=1e-6),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
        fit_config=fit_config,
    )
    baseline_config = torch.zeros(p, dtype=torch.float64)
    candidate_config = torch.ones(p, dtype=torch.float64)
    times = []
    for i in range(repeats):
        gp = GPSurrogate(
            X, Y, config=cfg, cat_dims=[], log_trafo_dims=[], bounds=None,
            shapley_configs=(baseline_config, candidate_config),
        )
        start = time.perf_counter()
        gp.fit()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  [{label}] rep {i}: {elapsed:.2f}s "
              f"lengthscale_mean={gp._model.covar_module.base_kernel.lengthscale.mean().item():.3f}",
              flush=True)
    return times


def main():
    run_dirs = {
        "resnet_14 (p=14)": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/24",
        "vit_16 (p=16)": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/90",
    }
    for name, run_dir in run_dirs.items():
        X, Y, bb = load_archive(run_dir)
        print(f"\n=== {name}: t={X.shape[0]}, blackbox={bb} ===", flush=True)

        print("-- exact (MLMConfig) --", flush=True)
        t_exact = time_fit(X, Y, MLMConfig(amount_restarts=5), "exact")

        print("-- accelerated, current default (fold + forced iterative) --", flush=True)
        t_fast_cur = time_fit(
            X, Y,
            AcceleratedFitConfig(amount_restarts=5, use_iterative=True,
                                  inducing_points=None, fold_odd_part=True),
            "fold+iter")

        print("-- accelerated, fold + DIRECT Cholesky (no forced iterative) --", flush=True)
        t_fast_direct = time_fit(
            X, Y,
            AcceleratedFitConfig(amount_restarts=5, use_iterative=False,
                                  inducing_points=None, fold_odd_part=True),
            "fold+direct")

        print(f"\n  summary: exact_mean={sum(t_exact)/len(t_exact):.2f}s  "
              f"fold+iter_mean={sum(t_fast_cur)/len(t_fast_cur):.2f}s  "
              f"fold+direct_mean={sum(t_fast_direct)/len(t_fast_direct):.2f}s", flush=True)


if __name__ == "__main__":
    main()
