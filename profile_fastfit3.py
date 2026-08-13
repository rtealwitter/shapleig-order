"""Follow-up to profile_fastfit2.py: exact converges in 229 fevals; the
folded auxiliary model needs 1389 to hit the same strict scipy L-BFGS-B
tolerance, on resnet_14 -- 6x more evals swamps the ~3-4x per-eval saving
from the halved matrix. Tests capping the auxiliary optimizer's maxiter/
maxfun directly (via fit_gpytorch_mll's optimizer_kwargs -> scipy
`options`), and checks how much the resulting hyperparameters (and the
downstream odd-part MLL value itself, as a cheap proxy for fit quality)
drift from the fully-converged optimum as the cap tightens.
"""

import json
import sys
import time

import torch

sys.path.insert(0, "shapleig-repo/src")

from xac.surrogates import (ConstantNoiseConfig, GPSurrogateConfig,
                             HammingKernelConfig)
from xac.surrogates.fast_fit import (OddKernel, _build_categorical_base,
                                      _complement_pairs, _find_categorical_kernel)
from botorch.models.gp_regression_mixed import SingleTaskGP
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

torch.set_default_dtype(torch.float64)


def load_archive(run_dir):
    with open(f"{run_dir}/metrics.json") as f:
        d = json.load(f)
    X = torch.tensor(d["archive_x"], dtype=torch.float64)
    Y = torch.tensor(d["archive_y"], dtype=torch.float64).reshape(-1, 1)
    return X, Y, d["blackbox"]


def build_folded_aux(train_X, train_Y, noise_level, min_ls):
    t, p = train_X.shape
    fold = _complement_pairs(train_X)
    i_idx, j_idx = fold[:, 0], fold[:, 1]
    fit_X = train_X[i_idx]
    fit_Y = 0.5 * (train_Y[i_idx] - train_Y[j_idx])
    base = OddKernel(_build_categorical_base(p, min_ls))
    covar_module = ScaleKernel(base)
    m = fit_X.shape[0]
    fit_Yvar = torch.full((m, 1), float(noise_level) * 0.5, dtype=fit_X.dtype)
    aux_model = SingleTaskGP(train_X=fit_X, train_Y=fit_Y, train_Yvar=fit_Yvar,
                              covar_module=covar_module)
    return aux_model, covar_module


def fit_with_cap(train_X, train_Y, maxiter, noise_level=1e-6, min_ls=1e-6):
    aux_model, covar_module = build_folded_aux(train_X, train_Y, noise_level, min_ls)
    mll = ExactMarginalLogLikelihood(aux_model.likelihood, aux_model)
    kwargs = {}
    if maxiter is not None:
        kwargs["optimizer_kwargs"] = {"options": {"maxiter": maxiter, "maxfun": maxiter + 20}}
    start = time.perf_counter()
    fit_gpytorch_mll(mll, max_attempts=5, pick_best_of_all_attempts=False, **kwargs)
    elapsed = time.perf_counter() - start
    ls = _find_categorical_kernel(covar_module).lengthscale.detach().clone()
    os_ = covar_module.outputscale.detach().clone()
    # fit_gpytorch_mll leaves the model in eval mode; evaluating mll there
    # runs the *posterior* path, not the training-likelihood path the
    # optimizer actually maximized, so the two aren't comparable -- switch
    # back to train mode first.
    aux_model.train()
    mll.train()
    with torch.no_grad():
        mll_value = mll(aux_model(aux_model.train_inputs[0]), aux_model.train_targets).sum().item()
    aux_model.eval()
    mll.eval()
    # Kernel-behavior check: raw lengthscale-vector distance is a poor
    # proxy when several ARD dims are underdetermined by the data (large,
    # loosely-constrained lengthscales that barely affect predictions) --
    # compare the actual training kernel matrices instead.
    with torch.no_grad():
        K = covar_module(train_X, train_X).to_dense()
    return elapsed, ls, os_, mll_value, K


def main():
    run_dirs = {
        "resnet_14": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/24",
        "vit_16": "shapleig-repo/multirun/2026-08-12/15-19-16/single_runs/90",
    }
    for name, run_dir in run_dirs.items():
        X, Y, bb = load_archive(run_dir)
        print(f"\n=== {bb}: t={X.shape[0]} ===", flush=True)

        print("-- fully converged (no cap) --", flush=True)
        e0, ls0, os0, mll0, K0 = fit_with_cap(X, Y, None)
        print(f"  time={e0:.2f}s outputscale={os0.item():.4f} mll={mll0:.4f}", flush=True)
        K0_norm = K0.norm().item()

        for cap in [400, 250, 150, 100, 60, 30]:
            e, ls, os_, mllv, K = fit_with_cap(X, Y, cap)
            ls_reldiff = (ls - ls0).norm().item() / ls0.norm().item()
            os_reldiff = abs(os_.item() - os0.item()) / abs(os0.item())
            mll_gap = mll0 - mllv  # >=0 if mll0 is truly the better optimum
            K_reldiff = (K - K0).norm().item() / K0_norm
            print(f"  cap={cap:4d}  time={e:6.2f}s  speedup={e0/e:5.2f}x  "
                  f"ls_reldiff={ls_reldiff:.4f}  os_reldiff={os_reldiff:.4f}  "
                  f"mll_gap={mll_gap:+.4f}  K_reldiff={K_reldiff:.4f}", flush=True)


if __name__ == "__main__":
    main()
