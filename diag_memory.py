"""Standalone diagnostic: does fit_accelerated leak memory across many
sequential calls within one process (mimicking one joblib worker's
lifecycle for a single (game, seed) run, t growing from ~11 to ~520)?

Logs ru_maxrss (MB, monotonic non-decreasing by definition on Linux) after
every Nth call so growth is visible directly, without 90-way parallelism
as a confound.
"""

import gc
import resource
import sys

import torch

sys.path.insert(0, "shapleig-repo/src")

from xac.surrogates import (AcceleratedFitConfig, ConstantNoiseConfig,
                             GPSurrogate, GPSurrogateConfig,
                             HammingKernelConfig)

torch.set_default_dtype(torch.float64)

P = 10
FULL = (1 << P) - 1


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def random_paired_archive(rng, n_pairs, n_extra):
    """n_pairs complement pairs plus n_extra random singles, like a
    PairedExtremes-then-EIG-ish mix."""
    masks = set()
    rows = []
    while len(rows) < 2 * n_pairs:
        s = rng.integers(0, 1 << P)
        c = FULL - s
        if s in masks or c in masks or s == c:
            continue
        masks.add(s)
        masks.add(c)
        rows.append(s)
        rows.append(c)
    while len(rows) < 2 * n_pairs + n_extra:
        s = rng.integers(0, 1 << P)
        if s in masks:
            continue
        masks.add(s)
        rows.append(s)
    bits = [[(m >> j) & 1 for j in range(P)] for m in rows]
    return torch.tensor(bits, dtype=torch.float64)


def main():
    import numpy as np

    rng = np.random.default_rng(0)
    cfg = GPSurrogateConfig(
        kernel_config=HammingKernelConfig(min_lengthscale=1e-6),
        noise_config=ConstantNoiseConfig(noise_level=1e-6),
        fit_config=AcceleratedFitConfig(),
    )

    baseline_config = torch.zeros(P, dtype=torch.float64)
    candidate_config = torch.ones(P, dtype=torch.float64)

    X = random_paired_archive(rng, n_pairs=5, n_extra=1)
    Y = torch.randn(X.shape[0], 1, dtype=torch.float64)
    gp = GPSurrogate(
        X, Y, config=cfg, cat_dims=[], log_trafo_dims=[], bounds=None,
        shapley_configs=(baseline_config, candidate_config),
    )
    gp.fit()
    print(f"[init] t={X.shape[0]:4d} rss={rss_mb():8.1f} MB", flush=True)

    n_pairs, n_extra = 5, 1
    for i in range(1, 321):
        # Grow by one point per step, alternating pair/single like a real run.
        if i % 3 == 0:
            n_extra += 1
        else:
            n_pairs += 1 if i % 2 == 0 else 0
            n_extra += 1 if i % 2 != 0 else 0
        X = random_paired_archive(rng, n_pairs, n_extra)
        Y = torch.randn(X.shape[0], 1, dtype=torch.float64)
        gp.update_data(X, Y)
        gp.fit()
        if i % 10 == 0:
            gc.collect()
            print(f"[{i:4d}] t={X.shape[0]:4d} rss={rss_mb():8.1f} MB", flush=True)
        if X.shape[0] >= 300:
            break

    print("done", flush=True)


if __name__ == "__main__":
    main()
