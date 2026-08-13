"""Integration check for max_optimizer_evals (see fast_fit.py docstring,
technique (6)): confirms the config wiring through GPSurrogate.fit() ->
fit_accelerated() reproduces the same speedup measured directly against
fit_gpytorch_mll in profile_fastfit3.py, on both the game where it should
matter (resnet_14) and the one where it shouldn't bind at all (vit_16).
"""

import json
import sys
import time

import torch

sys.path.insert(0, "shapleig-repo/src")

from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig, MLMConfig)

torch.set_default_dtype(torch.float64)


def load_archive(run_dir):
    with open(f"{run_dir}/metrics.json") as f:
        d = json.load(f)
    X = torch.tensor(d["archive_x"], dtype=torch.float64)
    Y = torch.tensor(d["archive_y"], dtype=torch.float64).reshape(-1, 1)
    return X, Y, d["blackbox"]


def time_fit(X, Y, fit_config, label, repeats=2):
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
        print(f"  [{label}] rep {i}: {elapsed:.2f}s", flush=True)
    return sum(times) / len(times)


def main():
    run_dirs = {
        "resnet_14 (p=14)": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/24",
        "vit_16 (p=16)": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/90",
        "dvbsrf_10 (p=10, hybrid, worst case)": "shapleig-repo/multirun/2026-08-12/17-16-54/single_runs/179",
        "vit_9 (p=9, hybrid, worst case)": "shapleig-repo/multirun/2026-08-12/15-19-14/single_runs/30",
    }
    for name, run_dir in run_dirs.items():
        X, Y, bb = load_archive(run_dir)
        print(f"\n=== {name}: t={X.shape[0]} ===", flush=True)

        print("-- exact --", flush=True)
        t_exact = time_fit(X, Y, MLMConfig(amount_restarts=5), "exact")

        print("-- accelerated, OLD default (no cap) --", flush=True)
        t_old = time_fit(
            X, Y,
            AcceleratedFitConfig(amount_restarts=5, use_iterative=True,
                                  inducing_points=None, fold_odd_part=True,
                                  max_optimizer_evals=None),
            "uncapped")

        print("-- accelerated, NEW default (max_optimizer_evals=250) --", flush=True)
        t_new = time_fit(
            X, Y,
            AcceleratedFitConfig(amount_restarts=5, use_iterative=True,
                                  inducing_points=None, fold_odd_part=True,
                                  max_optimizer_evals=250),
            "capped-250")

        print(f"\n  summary: exact={t_exact:.2f}s  old_fast(uncapped)={t_old:.2f}s  "
              f"new_fast(cap=250)={t_new:.2f}s  "
              f"new_vs_exact_speedup={t_exact/t_new:.2f}x  "
              f"new_vs_old_speedup={t_old/t_new:.2f}x", flush=True)


if __name__ == "__main__":
    main()
