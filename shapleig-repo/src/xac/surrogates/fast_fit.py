from __future__ import annotations

"""Faster hyperparameter fitting for the Hamming-kernel GP surrogate.

``GPSurrogate.fit()`` optimizes the exact marginal likelihood by Cholesky
factorizing the full ``t x t`` noisy training kernel at every evaluation,
``O(t^3)``. Nothing downstream needs that particular fitting procedure: the
ESP-based Shapley math in ``applications.py`` only ever reads the *fitted*
(lengthscale, outputscale) plus the exact archive off ``surrogate._model``
and recomputes the Shapley posterior itself. This module supplies a second
fitting procedure -- three standard accelerations, applied only to an
*auxiliary* model used purely to find good hyperparameters faster, whose
fitted (lengthscale, outputscale) are then copied onto the ordinary exact
model. Everything downstream of ``.fit()`` is unchanged.

1. **Iterative linear algebra.** ``gpytorch.settings.max_cholesky_size(0)``
   forces every marginal-likelihood evaluation to go through conjugate
   gradients plus stochastic Lanczos quadrature (for the log-determinant
   term and its gradient) instead of an exact Cholesky factorization --
   matrix-vector products instead of an ``O(t^3)`` factorization.
2. **Sparse inducing points.** ``gpytorch.kernels.InducingPointKernel`` (the
   standard Titsias/SGPR approximation) wraps the same Hamming kernel for
   the auxiliary fitting model only, reducing the per-evaluation cost to
   ``O(t m^2 + m^3)`` for ``m`` inducing points.
3. **Odd-part folding.** When the queried archive is exactly
   complement-paired (every coalition's complement was also queried --
   true by construction for the fixed paired-extremes schedule, and for the
   extremes phase of the hybrid schedule), the ``t`` observations fold into
   ``t/2`` observations of the odd part under the induced *odd kernel*
   ``k_odd(S, T) = 1/2 * (k(S, T) - k(S, T^c))``. The Shapley values only
   read the odd part of the value function (see
   ``shapleig_greedy_note.md``, "the even-odd decomposition in eigenbasis
   form"), so nothing the marginal likelihood should be fitting is lost,
   and every Cholesky/CG side length halves.

All three are independently toggleable and compose (folding first, then
inducing points on the folded archive), but measurement narrowed how (1)
and (2) actually get used here, versus how they're usually pitched:

- **(2) and (3) don't stack.** Both exist to dodge the same O(t^3)
  factorization, so once inducing points have already reduced the problem
  to an m x m system, forcing iterative solves on top only adds CG/Lanczos
  overhead for a system already cheap to factor directly.
- **Inducing points are disabled by default** (``inducing_points=None``):
  at the archive sizes this project's sweeps reach (t up to ~500),
  ``InducingPointKernel``'s own construction/evaluation overhead measured
  *slower* than a direct factorization -- the O(t^3) cost was never the
  bottleneck to begin with here. Left in, documented, for a scale where
  that trade flips.
- **Iterative solves are forced only when folding is active.** Forcing them
  on a large, un-reduced archive (e.g. the hybrid arm well past its
  extremes handover, where folding has self-disabled because the archive
  is mostly unpaired EIG-selected points) OOM'd a 90-worker sweep in
  practice -- gpytorch's CG/Lanczos path is not memory-lean at t~500 on the
  full kernel. On an unfolded archive this module falls back to an
  ordinary direct fit: the same cost the exact baseline already pays
  successfully at that size, so no acceleration but no regression either.

Each technique falls back gracefully rather than raising: folding is
skipped when the archive is not complement-paired, and the whole
accelerated routine is wrapped by its caller (``GPSurrogate.fit()``) so any
numerical failure here falls through to the ordinary exact fit instead of
crashing an unattended sweep.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from .gp_surrogate import ConstantNoiseConfig, MLMConfig

log = logging.getLogger(__name__)

__all__ = ["AcceleratedFitConfig", "OddKernel", "fit_accelerated"]

# Folding requires at least this fraction of the archive covered by
# complement pairs (see _fit_accelerated_inner).
_MIN_PAIR_COVERAGE = 0.8


@dataclass(frozen=True)
class AcceleratedFitConfig(MLMConfig):
    """``MLMConfig`` that fits (lengthscale, outputscale) faster; see the
    module docstring for the three techniques.

    ``inducing_points`` defaults to ``None`` (disabled): at the archive
    sizes this project's sweeps actually reach (t up to ~500), measurement
    showed ``InducingPointKernel``'s own module-construction and
    per-evaluation overhead exceeds what it saves over a direct factorization
    -- the O(t^3) factorization was never the dominant cost here to begin
    with. It is left in as a correct, validated-not-to-crash option for
    larger archives where that trade flips; set it explicitly to use it."""

    use_iterative: bool = True  # (2) CG + Lanczos via max_cholesky_size(0)
    inducing_points: Optional[int] = None  # (3) m << t; None disables (see above)
    fold_odd_part: bool = True  # (5) complement-paired archives only


class OddKernel(Kernel):
    r"""``k_odd(S, T) = 1/2 (k(S, T) - k(S, T^c))`` for a product kernel
    ``k`` over binary coalition vectors, with ``T^c = 1 - T`` under 0/1
    encoding. This is exactly the covariance of the odd part
    ``nu_odd(S) = 1/2 (nu(S) - nu(S^c))`` of any function with a
    ``k``-covariance GP prior, since covariance is bilinear."""

    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        k_same = self.base_kernel(x1, x2, diag=diag, **params)
        k_comp = self.base_kernel(x1, 1.0 - x2, diag=diag, **params)
        if hasattr(k_same, "to_dense"):
            k_same = k_same.to_dense()
        if hasattr(k_comp, "to_dense"):
            k_comp = k_comp.to_dense()
        return 0.5 * (k_same - k_comp)


def _build_categorical_base(p: int, min_lengthscale: float) -> CategoricalKernel:
    """Same ARD Hamming-kernel construction as ``GPSurrogate``'s
    ``_build_base_kernel`` for ``HammingKernelConfig``, minus the Kronecker
    full-grid shortcut (never triggered on a training archive with
    ``t << 2^p`` rows, so mathematically identical here)."""
    sqrt2, sqrt3 = math.sqrt(2), math.sqrt(3)
    lengthscale_prior = LogNormalPrior(loc=sqrt2 + math.log(p) * 0.5, scale=sqrt3)
    return CategoricalKernel(
        ard_num_dims=p,
        lengthscale_prior=lengthscale_prior,
        lengthscale_constraint=GreaterThan(
            min_lengthscale,
            initial_value=max(lengthscale_prior.mode, min_lengthscale),
        ),
    )


def _find_categorical_kernel(module: Kernel) -> CategoricalKernel:
    """Walk ``.base_kernel`` wrappers (``ScaleKernel``, ``InducingPointKernel``,
    ``OddKernel``) down to the underlying ``CategoricalKernel``."""
    if isinstance(module, CategoricalKernel):
        return module
    if hasattr(module, "base_kernel"):
        return _find_categorical_kernel(module.base_kernel)
    msg = f"No CategoricalKernel found inside {type(module)}."
    raise RuntimeError(msg)


def _complement_pairs(archive_X_bin: torch.Tensor) -> Optional[torch.Tensor]:
    """``(m, 2)`` row-index pairs ``(i, j)`` with row ``j`` the bitwise
    complement of row ``i``, one pair per matched coalition; ``None`` if no
    pair exists at all. A leftover coalition without a partner (e.g. the odd
    point out of a size-``p+1`` initial design, which cannot perfectly pair
    for even ``p``) is simply left out of the returned pairs rather than
    blocking folding for the whole archive -- dropping one point out of a
    growing archive costs the auxiliary fit almost nothing. A duplicate
    coalition mask keeps only its first occurrence as a pairing candidate."""
    t, p = archive_X_bin.shape
    bits = (archive_X_bin > 0.5).to(torch.int64)
    weights = 2 ** torch.arange(p, dtype=torch.int64, device=bits.device)
    masks = (bits * weights).sum(dim=1).tolist()

    mask_to_idx: dict[int, int] = {}
    for idx, m in enumerate(masks):
        mask_to_idx.setdefault(m, idx)

    full = (1 << p) - 1
    seen: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for m, idx in mask_to_idx.items():
        if m in seen:
            continue
        comp = full - m
        if comp not in mask_to_idx or comp == m:
            continue
        pairs.append((idx, mask_to_idx[comp]))
        seen.add(m)
        seen.add(comp)
    if not pairs:
        return None
    return torch.tensor(pairs, dtype=torch.long, device=archive_X_bin.device)


def _inducing_subset(X: torch.Tensor, m: int) -> torch.Tensor:
    """``m`` evenly-spaced rows of ``X`` (a fixed, deterministic subsample
    rather than a continuously-optimized location -- the input domain is
    the binary hypercube, so "moving" an inducing point off the training
    rows would leave the input space the kernel is defined on)."""
    n = X.shape[0]
    idx = torch.linspace(0, n - 1, m).round().long().unique()
    return X[idx].clone()


def fit_accelerated(surrogate) -> None:
    """Fit ``surrogate._model``'s (lengthscale, outputscale) via the
    accelerated auxiliary-model route and copy them onto the real model.
    Raises on any failure; the caller (``GPSurrogate.fit()``) falls back to
    the ordinary exact fit."""
    cfg = surrogate.config.fit_config
    assert isinstance(cfg, AcceleratedFitConfig)
    model = surrogate._model

    torch_default_dtype = torch.get_default_dtype()
    try:
        # Constructing under float64 default matters here for the same
        # reason as HammingGP._build: torch creates the prior/constraint
        # buffers and initial values in the default dtype before botorch
        # casts to the (float64) data, and a float32 default truncates
        # them, silently perturbing the fitted hyperparameters.
        torch.set_default_dtype(torch.float64)
        _fit_accelerated_inner(surrogate, cfg, model)
    finally:
        torch.set_default_dtype(torch_default_dtype)


def _fit_accelerated_inner(surrogate, cfg: AcceleratedFitConfig, model) -> None:
    train_X = model.train_inputs[0].detach().clone()
    train_Y = model.train_targets.detach().clone().reshape(-1, 1)
    t, p = train_X.shape
    min_ls = surrogate.config.kernel_config.min_lengthscale
    noise_is_fixed = isinstance(surrogate.config.noise_config, ConstantNoiseConfig)

    fold = _complement_pairs(train_X) if cfg.fold_odd_part else None
    # Fold only when pairs cover most of the archive: a handful of
    # incidental complements inside a mostly-unpaired (e.g. EIG-selected)
    # archive isn't worth it -- the auxiliary fit would drop most of the
    # archive's information along with the unpaired majority. This also
    # makes folding self-disable for the hybrid arm once its EIG phase
    # dilutes the fixed-size paired-extremes prefix.
    used_fold = (
        fold is not None
        and fold.shape[0] >= 2
        and 2 * fold.shape[0] >= _MIN_PAIR_COVERAGE * t
    )

    if used_fold:
        i_idx, j_idx = fold[:, 0], fold[:, 1]
        fit_X = train_X[i_idx]
        fit_Y = 0.5 * (train_Y[i_idx] - train_Y[j_idx])
        base = OddKernel(_build_categorical_base(p, min_ls))
    else:
        fit_X, fit_Y = train_X, train_Y
        base = _build_categorical_base(p, min_ls)

    m = fit_X.shape[0]
    covar_base = base
    used_inducing = cfg.inducing_points is not None and m > cfg.inducing_points
    if used_inducing:
        Z = _inducing_subset(fit_X, cfg.inducing_points)
        covar_base = InducingPointKernel(
            base, inducing_points=Z, likelihood=GaussianLikelihood()
        )
    covar_module = ScaleKernel(covar_base)

    if noise_is_fixed:
        noise_scalar = model.likelihood.noise.detach().reshape(-1)[0]
        fold_scale = 0.5 if used_fold else 1.0
        fit_Yvar = torch.full(
            (m, 1), float(noise_scalar) * fold_scale, dtype=fit_X.dtype
        )
        aux_model = SingleTaskGP(
            train_X=fit_X, train_Y=fit_Y, train_Yvar=fit_Yvar, covar_module=covar_module
        )
    else:
        aux_model = SingleTaskGP(
            train_X=fit_X,
            train_Y=fit_Y,
            covar_module=covar_module,
            likelihood=GaussianLikelihood(),
        )

    mll = ExactMarginalLogLikelihood(aux_model.likelihood, aux_model)
    # Iterative solves (2) and inducing points (3) both exist to dodge the
    # same O(t^3) factorization, so they don't stack: once InducingPointKernel
    # has already reduced the system to an m x m (SGPR/Titsias) problem,
    # forcing CG/Lanczos on top adds iterative-solver overhead for no
    # benefit -- m is already small enough for a direct solve to be cheap.
    #
    # Also only force it when odd-part folding is active: measurement (a
    # HybridPairedEIG run well past its extremes handover, archive mostly
    # unpaired at t~500, used_fold=False) OOM'd a 200GB/90-worker job --
    # gpytorch's CG/Lanczos path is not the lean option its name suggests at
    # this scale on a full, un-reduced archive. Direct Cholesky at t~500 is
    # exactly what the exact baseline already does successfully, so falling
    # back to it here (no acceleration, but no regression either) is safe;
    # forcing iterative solves is validated safe and worthwhile only on the
    # already-halved folded system.
    force_iterative = cfg.use_iterative and not used_inducing and used_fold
    cholesky_size = 0 if force_iterative else gpytorch.settings.max_cholesky_size.value()
    with gpytorch.settings.max_cholesky_size(cholesky_size):
        fit_gpytorch_mll(
            mll, max_attempts=cfg.amount_restarts, pick_best_of_all_attempts=False
        )

    fitted_base = _find_categorical_kernel(covar_module)
    lengthscale = fitted_base.lengthscale.detach().clone()
    outputscale = covar_module.outputscale.detach().clone()

    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale = lengthscale
        model.covar_module.outputscale = outputscale
        if not noise_is_fixed:
            fold_scale = 2.0 if used_fold else 1.0
            model.likelihood.noise = aux_model.likelihood.noise.detach() * fold_scale

    model.eval()
    log.info(
        "Accelerated fit: t=%d, folded=%s (m=%d), inducing=%s, iterative=%s",
        t,
        used_fold,
        m,
        cfg.inducing_points if used_inducing else None,
        force_iterative,
    )
