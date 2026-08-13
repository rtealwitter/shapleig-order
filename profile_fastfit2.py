"""Follow-up to profile_fastfit.py: the first pass ruled out CG/Lanczos
overhead (fold+direct-Cholesky was exactly as slow as fold+iterative on
resnet_14). This instruments *how many times* and *how long* the
scipy L-BFGS-B optimizer inside fit_gpytorch_mll actually runs, to check
whether the auxiliary (odd-part-folded) model's likelihood surface just
needs many more restart attempts / iterations to converge than the exact
model's, on resnet_14 specifically.
"""

import json
import sys
import time
import warnings

import scipy.optimize
import torch

sys.path.insert(0, "shapleig-repo/src")

from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig, MLMConfig)

torch.set_default_dtype(torch.float64)

_orig_minimize = scipy.optimize.minimize
_calls = []


def _tracking_minimize(*args, **kwargs):
    start = time.perf_counter()
    res = _orig_minimize(*args, **kwargs)
    _calls.append({
        "elapsed": time.perf_counter() - start,
        "nit": getattr(res, "nit", None),
        "nfev": getattr(res, "nfev", None),
        "success": getattr(res, "success", None),
        "message": str(getattr(res, "message", "")),
    })
    return res


scipy.optimize.minimize = _tracking_minimize
# botorch.optim.utils.timeout.minimize_with_timeout does `from scipy import
# optimize` then calls `optimize.minimize(...)`, so it resolves the
# attribute at call time -- patching scipy.optimize.minimize is enough,
# no need to patch a botorch-local reference.


def load_archive(run_dir):
    with open(f"{run_dir}/metrics.json") as f:
        d = json.load(f)
    X = torch.tensor(d["archive_x"], dtype=torch.float64)
    Y = torch.tensor(d["archive_y"], dtype=torch.float64).reshape(-1, 1)
    return X, Y, d["blackbox"]


def time_fit(X, Y, fit_config, label):
    global _calls
    p = X.shape[1]
    cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=1e-6),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
        fit_config=fit_config,
    )
    baseline_config = torch.zeros(p, dtype=torch.float64)
    candidate_config = torch.ones(p, dtype=torch.float64)
    gp = GPSurrogate(
        X, Y, config=cfg, cat_dims=[], log_trafo_dims=[], bounds=None,
        shapley_configs=(baseline_config, candidate_config),
    )
    _calls = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        gp.fit()
        elapsed = time.perf_counter() - start
    print(f"  [{label}] total={elapsed:.2f}s  scipy_minimize_calls={len(_calls)}", flush=True)
    for i, c in enumerate(_calls):
        print(f"    call {i}: {c['elapsed']:.2f}s nit={c['nit']} nfev={c['nfev']} "
              f"success={c['success']} msg={c['message'][:60]!r}", flush=True)
    warn_msgs = [str(w.message)[:80] for w in caught]
    if warn_msgs:
        print(f"    warnings ({len(warn_msgs)}): {warn_msgs[:5]}", flush=True)
    return elapsed


def main():
    run_dirs = {
        "resnet_14 (p=14)": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/24",
    }
    for name, run_dir in run_dirs.items():
        X, Y, bb = load_archive(run_dir)
        print(f"\n=== {name}: t={X.shape[0]}, blackbox={bb} ===", flush=True)

        print("-- exact (MLMConfig) --", flush=True)
        time_fit(X, Y, MLMConfig(amount_restarts=5), "exact")

        print("-- accelerated, fold + direct Cholesky --", flush=True)
        time_fit(
            X, Y,
            AcceleratedFitConfig(amount_restarts=5, use_iterative=False,
                                  inducing_points=None, fold_odd_part=True),
            "fold+direct")


if __name__ == "__main__":
    main()
