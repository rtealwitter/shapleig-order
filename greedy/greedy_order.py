# Exhaustive greedy ShaplEIG design with a FIXED kernel: no observations y
# ever enter, so the whole sequence is a function of (d, beta, sigma2).
# For small d we materialize the full 2^d posterior and run greedy argmax-EIG,
# recording the chosen coalition, its size, tie-class structure, and verifying
# that scores are constant on orbits of the symmetry group.
import numpy as np
from math import comb
from collections import Counter, defaultdict

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def membership(d):
    N = 1 << d
    masks = np.arange(N)
    return ((masks[:, None] >> np.arange(d)) & 1).astype(bool)


def kernel_matrix(Mem, betas):
    N, d = Mem.shape
    L = np.zeros((N, N))
    for j in range(d):
        mj = Mem[:, j]
        L += np.log(betas[j]) * (mj[:, None] ^ mj[None, :])
    return np.exp(L)


def shapley_matrix(Mem):
    N, d = Mem.shape
    sizes = Mem.sum(1)
    A = np.zeros((d, N))
    for i in range(d):
        inS = Mem[:, i]
        w = np.empty(N)
        for S in range(N):
            s = sizes[S]
            if inS[S]:
                w[S] = 1.0 / (d * comb(d - 1, s - 1))
            else:
                w[S] = -1.0 / (d * comb(d - 1, s))
        A[i] = w
    return A


def scores(K, A, E, sigma2):
    """rho^2 for every coalition given evaluated index list E."""
    N = K.shape[0]
    if E:
        KEE = K[np.ix_(E, E)] + sigma2 * np.eye(len(E))
        KxE = K[:, E]
        Sigma = K - KxE @ np.linalg.solve(KEE, KxE.T)
    else:
        Sigma = K.copy()
    ASig = A @ Sigma                      # d x N ; column T is c_T
    G = ASig @ A.T                        # A Sigma A^T
    G = 0.5 * (G + G.T)
    Ginv = np.linalg.pinv(G, rcond=1e-11, hermitian=True)
    v = np.maximum(np.diag(Sigma), 0.0) + sigma2
    num = np.einsum('it,ij,jt->t', ASig, Ginv, ASig)
    return num / v


def atom_profile_all(Mem, E):
    """For each candidate mask, its orbit key: counts of taken/total per atom."""
    N, d = Mem.shape
    # atom of player j = pattern of membership of j across evaluated sets
    pats = [tuple(bool(Mem[e][j]) for e in E) for j in range(d)]
    keys = []
    for S in range(N):
        cnt = Counter()
        for j in range(d):
            if Mem[S, j]:
                cnt[pats[j]] += 1
        keys.append(tuple(sorted(cnt.items())))
    return keys


def run(d, betas, sigma2=1e-6, steps=None, label="", check_orbits=True):
    betas = np.asarray(betas, float)
    steps = steps or min(3 * d, 30)
    Mem = membership(d)
    N = 1 << d
    sizes = Mem.sum(1)
    K = kernel_matrix(Mem, betas)
    A = shapley_matrix(Mem)
    E = []
    print(f"\n===== d={d}  betas={label or betas}  sigma2={sigma2} =====")
    r0 = scores(K, A, E, sigma2)
    by_size = {s: r0[sizes == s].mean() for s in range(d + 1)}
    spread = {s: np.ptp(r0[sizes == s]) for s in range(d + 1)}
    print("step-0 rho^2 by size:")
    for s in range(d + 1):
        print(f"  |T|={s:2d}: mean={by_size[s]:.9f}  within-size spread={spread[s]:.2e}")
    for step in range(steps):
        r = scores(K, A, E, sigma2)
        r[E] = -np.inf
        best = r.max()
        tol = 1e-8 * max(1.0, abs(best))
        tied = np.where(r >= best - tol)[0]
        tied_sizes = Counter(int(sizes[t]) for t in tied)
        # orbit check: is the tie class a union of orbits, and score constant on orbits?
        orbit_note = ""
        if check_orbits:
            keys = atom_profile_all(Mem, E)
            groups = defaultdict(list)
            for S in range(N):
                if S not in E:
                    groups[keys[S]].append(S)
            worst = max(np.ptp(r[np.array(g)]) for g in groups.values())
            top_orbits = len({keys[t] for t in tied})
            orbit_note = f" | orbit-score spread max={worst:.1e}, tie covers {top_orbits} orbit(s)"
        # deterministic tie-break: smallest size, then smallest mask
        chosen = min(tied.tolist(), key=lambda S: (sizes[S], S))
        pset = tuple(int(j) for j in range(d) if Mem[chosen, j])
        inter = [len(set(pset) & {j for j in range(d) if Mem[e, j]}) for e in E]
        print(f"step {step:2d}: pick {pset} size={sizes[chosen]} rho2={best:.6f} "
              f"ties={len(tied)} tie-sizes={dict(tied_sizes)} |cap prev|={inter}{orbit_note}")
        E.append(int(chosen))
    # pairwise symmetric-difference distances of the chosen design
    print("pairwise |S_a xor S_b|:")
    D = np.zeros((len(E), len(E)), int)
    for a in range(len(E)):
        for b in range(len(E)):
            D[a, b] = bin(E[a] ^ E[b]).count("1")
    print(D)
    return E


if __name__ == "__main__":
    for d in (6, 8):
        for beta in (0.3, 0.6, 0.9):
            run(d, [beta] * d, steps=2 * d, label=f"const {beta}")
    run(10, [0.6] * 10, steps=25, label="const 0.6")
    run(11, [0.6] * 11, steps=20, label="const 0.6", check_orbits=False)
    # asymmetric lengthscales: do picks track player relevance?
    d = 8
    bet = np.linspace(0.15, 0.95, d)
    run(d, bet, steps=2 * d, label=f"linspace {bet.round(2).tolist()}", check_orbits=False)
    # different noise level
    run(8, [0.6] * 8, sigma2=1e-2, steps=16, label="const 0.6, big noise")
