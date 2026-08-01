# Precompute shapiq local-XAI games as full value tables.
# For each game: 8 noisy tables (MarginalImputer with sample_size=128, one RNG
# seed each — the realistic Monte-Carlo noise in these games) plus one
# high-sample ground-truth table, from which the exact Shapley vector is
# computed densely. Resumable: skips games whose npz already exists.
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "greedy"))
from gp_core import membership, shapley_A

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
os.makedirs(OUT, exist_ok=True)
N_SEEDS = 8
SAMPLE_SIZE = 128
GT_SAMPLE = {8: 4096, 12: 2048, 14: 1024}


def eval_all(imputer, d, chunk=512):
    Mem = membership(d)
    N = Mem.shape[0]
    vals = np.empty(N)
    for lo in range(0, N, chunk):
        vals[lo:lo + chunk] = imputer(Mem[lo:lo + chunk])
    return vals


class ProbaWrap:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        return self.model.predict_proba(X)[:, 1]


def build_games():
    import shapiq
    from shapiq.datasets import load_california_housing, load_bike_sharing, load_adult_census
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    games = []

    Xc, yc = load_california_housing(to_numpy=True)
    Xtr, Xte, ytr, _ = train_test_split(Xc, yc, test_size=0.2, random_state=0)
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(Xtr, ytr)
    bg_c = Xtr[np.random.default_rng(0).choice(len(Xtr), 512, replace=False)]
    games.append(("california_i3", 8, gbr.predict, bg_c, Xte[3:4]))
    games.append(("california_i17", 8, gbr.predict, bg_c, Xte[17:18]))

    Xb, yb = load_bike_sharing(to_numpy=True)
    Xtr, Xte, ytr, _ = train_test_split(Xb, yb, test_size=0.2, random_state=0)
    gbr_b = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0).fit(Xtr, ytr)
    bg_b = Xtr[np.random.default_rng(0).choice(len(Xtr), 512, replace=False)]
    games.append(("bike_i5", 12, gbr_b.predict, bg_b, Xte[5:6]))

    Xa, ya = load_adult_census(to_numpy=True)
    Xtr, Xte, ytr, _ = train_test_split(Xa, ya, test_size=0.2, random_state=0)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    bg_a = Xtr[np.random.default_rng(0).choice(len(Xtr), 512, replace=False)]
    games.append(("adult_i7", 14, ProbaWrap(rf), bg_a, Xte[7:8]))

    for name, d, model, bg, x in games:
        path = os.path.join(OUT, f"{name}.npz")
        if os.path.exists(path):
            print(f"skip {name} (exists)")
            continue
        t0 = time.time()
        vals_seeds = []
        for s in range(N_SEEDS):
            imp = shapiq.MarginalImputer(model, bg, x=x, sample_size=SAMPLE_SIZE,
                                         normalize=False, random_state=s)
            vals_seeds.append(eval_all(imp, d))
            print(f"  {name} seed {s} done ({time.time() - t0:.0f}s)")
        imp_gt = shapiq.MarginalImputer(model, bg, x=x, sample_size=GT_SAMPLE[d],
                                        normalize=False, random_state=12345)
        vals_gt = eval_all(imp_gt, d)
        A = shapley_A(d)
        phi_star = A @ vals_gt
        # independent naive cross-check of the A matrix on the smallest games
        if d == 8:
            Mem = membership(d)
            from math import comb, factorial
            phi_naive = np.zeros(d)
            for i in range(d):
                for S in range(1 << d):
                    if not Mem[S, i]:
                        s = int(Mem[S].sum())
                        w = factorial(s) * factorial(d - s - 1) / factorial(d)
                        phi_naive[i] += w * (vals_gt[S | (1 << i)] - vals_gt[S])
            assert np.allclose(phi_naive, phi_star, atol=1e-10), "Shapley cross-check"
            print(f"  {name}: naive Shapley cross-check passed")
        noise = np.std([v - vals_gt for v in vals_seeds])
        np.savez(path, d=d, values_seeds=np.array(vals_seeds), values_gt=vals_gt,
                 phi_star=phi_star)
        print(f"{name}: d={d} phi*={np.round(phi_star, 4)} "
              f"per-eval noise sd ~ {noise:.5f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    build_games()
