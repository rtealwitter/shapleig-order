# Poly-time exact greedy ShaplEIG design for equal lengthscales (beta_j = beta).
# No 2^d objects: A k_T via the univariate generating polynomial, A K A^T in
# closed form from the parity basis, and candidates enumerated one per orbit of
# the design's stabilizer. Validates against the dense brute force at d<=10,
# then runs d=30 and d=100 where the lattice is astronomically out of reach.
import numpy as np
import time
from math import comb
from collections import defaultdict

np.set_printoptions(precision=6, suppress=True, linewidth=200)


# ---------- closed-form prior Shapley covariance  G = A K A^T ----------
# In the parity basis chi_U(S) = (-1)^{|U cap S|}, the Hamming kernel with
# equal beta has independent Fourier coefficients with variance
# 2^{-d} (1-beta)^u (1+beta)^{d-u}, u = |U|, and phi_i(chi_U) = -2/u for
# i in U with u odd, else 0. Hence G is exchangeable with
#   G_ii  = sum_{u odd} (4/u^2) C(d-1,u-1) 2^{-d} lam_u
#   G_ii' = sum_{u odd} (4/u^2) C(d-2,u-2) 2^{-d} lam_u.
def prior_shapley_cov(d, beta):
    diag = off = 0.0
    for u in range(1, d + 1, 2):
        lam = (1 - beta) ** u * (1 + beta) ** (d - u) / 2 ** d
        diag += 4.0 / u ** 2 * comb(d - 1, u - 1) * lam
        if u >= 2 or True:
            off += 4.0 / u ** 2 * (comb(d - 2, u - 2) if u >= 2 else 0) * lam
    return diag, off


# ---------- A k_T via the generating polynomial ----------
# Equal beta: the coefficients c_s = sum_{|S|=s} k(S,T) depend only on t=|T|,
# and [A k_T]_i takes just two values (i in T / i not in T). Returns those two
# values for every t in 0..d, each in O(d^2), O(d^3) total.
def AkT_table(d, beta):
    wplus = np.array([1.0 / (d * comb(d - 1, s - 1)) if s >= 1 else 0.0 for s in range(d + 1)])
    wminus = np.array([1.0 / (d * comb(d - 1, s)) if s <= d - 1 else 0.0 for s in range(d + 1)])
    def poly(n_in, n_out):
        # coefficients of (beta + z)^n_in (1 + beta z)^n_out, built as a
        # product; never divide (synthetic division amplifies error by beta^-d)
        c = np.zeros(n_in + n_out + 1)
        c[0] = 1.0
        for _ in range(n_in):
            c[1:] = beta * c[1:] + c[:-1].copy()
            c[0] = beta * c[0]
        for _ in range(n_out):
            c[1:] = c[1:] + beta * c[:-1].copy()
        return c

    val_in = np.zeros(d + 1)
    val_out = np.zeros(d + 1)
    for t in range(d + 1):
        c = poly(t, d - t)
        if t >= 1:
            q = poly(t - 1, d - t)            # leave out one (beta + z) factor
            in_sums = np.zeros(d + 1)
            in_sums[1:] = 1.0 * q             # membership restored with delta=1
            val_in[t] = np.sum(wplus[1:] * in_sums[1:]) - np.sum(wminus[:-1] * (c[:-1] - in_sums[:-1]))
        if t <= d - 1:
            q = poly(t, d - t - 1)            # leave out one (1 + beta z) factor
            in_sums = np.zeros(d + 1)
            in_sums[1:] = beta * q
            val_out[t] = np.sum(wplus[1:] * in_sums[1:]) - np.sum(wminus[:-1] * (c[:-1] - in_sums[:-1]))
    return val_in, val_out


# ---------- orbit enumeration ----------
# Atom signature of player j: (is {j} evaluated, is [d]\{j} evaluated,
# membership of j in every evaluated "middle" set). Any permutation preserving
# atoms stabilizes the design as a set, so scores are constant on profiles
# (how many players a candidate takes from each atom).
def candidate_reps(d, E_sets):
    middles = [S for S in E_sets if 1 < len(S) < d - 1]
    singles = {next(iter(S)) for S in E_sets if len(S) == 1}
    cosingles = {(set(range(d)) - S).pop() for S in E_sets if len(S) == d - 1}
    sig = {}
    for j in range(d):
        sig[j] = (j in singles, j in cosingles, tuple(j in M for M in middles))
    atoms = defaultdict(list)
    for j in range(d):
        atoms[sig[j]].append(j)
    atom_lists = list(atoms.values())
    total = 1
    for pool in atom_lists:
        total *= len(pool) + 1
    if total > 200000:
        return None  # orbit structure too fine; caller stops gracefully
    reps = []
    def rec(a, chosen):
        if a == len(atom_lists):
            reps.append(frozenset(chosen))
            return
        pool = atom_lists[a]
        for k in range(len(pool) + 1):
            rec(a + 1, chosen + pool[:k])
    rec(0, [])
    return [T for T in set(reps) if T not in E_sets]


# ---------- one greedy step, all candidates scored in poly time ----------
def greedy_run(d, beta, sigma2=1e-6, steps=None, verbose=True):
    steps = steps or 2 * d + 8
    diag, off = prior_shapley_cov(d, beta)
    G_prior = off * np.ones((d, d)) + (diag - off) * np.eye(d)
    val_in, val_out = AkT_table(d, beta)

    def Ak(T):
        t = len(T)
        v = np.full(d, val_out[t])
        v[list(T)] = val_in[t]
        return v

    E_sets, E_bool = [], np.zeros((0, d), bool)
    seq = []
    t0 = time.time()
    for step in range(steps):
        m = len(E_sets)
        if m:
            ham = (E_bool[:, None, :] ^ E_bool[None, :, :]).sum(-1)
            M = beta ** ham + sigma2 * np.eye(m)
            W = np.stack([Ak(S) for S in E_sets], axis=1)      # d x m
            Minv = np.linalg.inv(M)
            G = G_prior - W @ Minv @ W.T
        else:
            W = np.zeros((d, 0)); Minv = np.zeros((0, 0)); G = G_prior
        G = 0.5 * (G + G.T)
        Ginv = np.linalg.pinv(G, rcond=1e-11, hermitian=True)

        reps = candidate_reps(d, set(E_sets))
        if reps is None:
            print(f"step {step:3d}: stopping — orbit count exceeds cap")
            break
        Tb = np.zeros((len(reps), d), bool)
        for r, T in enumerate(reps):
            Tb[r, list(T)] = True
        tsz = Tb.sum(1)
        # k_m(T): beta^{|T xor S_a|}
        if m:
            ham_T = (Tb[:, None, :] ^ E_bool[None, :, :]).sum(-1)  # ncand x m
            kmT = beta ** ham_T
            sol = kmT @ Minv                                        # ncand x m
            v = 1.0 + sigma2 - np.einsum('cm,cm->c', sol, kmT)
            C = np.stack([Ak(T) for T in reps], axis=1) - W @ sol.T # d x ncand
        else:
            v = np.full(len(reps), 1.0 + sigma2)
            C = np.stack([Ak(T) for T in reps], axis=1)
        num = np.einsum('ic,ij,jc->c', C, Ginv, C)
        rho2 = num / np.maximum(v, 1e-15)
        best = rho2.max()
        tol = 1e-8 * max(1.0, best)
        tied = [reps[r] for r in np.where(rho2 >= best - tol)[0]]
        chosen = min(tied, key=lambda T: (len(T), sorted(T)))
        kind = ("empty" if len(chosen) == 0 else "full" if len(chosen) == d
                else f"singleton {min(chosen)}" if len(chosen) == 1
                else f"co-singleton {(set(range(d)) - chosen).pop()}" if len(chosen) == d - 1
                else f"middle size {len(chosen)} {sorted(chosen) if d <= 30 else ''}")
        seq.append((chosen, best))
        if verbose:
            szs = sorted({len(T) for T in tied})
            print(f"step {step:3d}: {kind:18s} rho2={best:.6f} cands={len(reps):5d} tie-sizes={szs}")
        E_sets.append(chosen)
        E_bool = np.vstack([E_bool, Tb[reps.index(chosen)]])
    print(f"[d={d} beta={beta}] {steps} steps in {time.time() - t0:.1f}s")
    return seq


# ---------- validation against dense brute force ----------
def brute_scores(d, beta, E_sets, sigma2=1e-6):
    N = 1 << d
    Mem = ((np.arange(N)[:, None] >> np.arange(d)) & 1).astype(bool)
    sizes = Mem.sum(1)
    L = np.zeros((N, N))
    for j in range(d):
        mj = Mem[:, j]
        L += np.log(beta) * (mj[:, None] ^ mj[None, :])
    K = np.exp(L)
    A = np.zeros((d, N))
    for i in range(d):
        for S in range(N):
            s = sizes[S]
            A[i, S] = (1.0 / (d * comb(d - 1, s - 1)) if Mem[S, i]
                       else -1.0 / (d * comb(d - 1, s)))
    E = [sum(1 << j for j in S) for S in E_sets]
    if E:
        KEE = K[np.ix_(E, E)] + sigma2 * np.eye(len(E))
        KxE = K[:, E]
        Sigma = K - KxE @ np.linalg.solve(KEE, KxE.T)
    else:
        Sigma = K
    ASig = A @ Sigma
    G = ASig @ A.T
    Ginv = np.linalg.pinv(0.5 * (G + G.T), rcond=1e-11, hermitian=True)
    v = np.maximum(np.diag(Sigma), 0) + sigma2
    return np.einsum('it,ij,jt->t', ASig, Ginv, ASig) / v, Mem


def validate(d, beta):
    print(f"--- validate d={d} beta={beta} ---")
    seq = greedy_run(d, beta, steps=min(2 * d + 4, 14), verbose=False)
    designs = [[], [s for s, _ in seq[:3]], [s for s, _ in seq[:7]]]
    worst = 0.0
    for E_sets in designs:
        bs, Mem = brute_scores(d, beta, E_sets)
        # efficient scorer on the same design, checked on every coalition:
        diag, off = prior_shapley_cov(d, beta)
        G_prior = off * np.ones((d, d)) + (diag - off) * np.eye(d)
        val_in, val_out = AkT_table(d, beta)
        def Ak(T):
            v = np.full(d, val_out[len(T)]); v[list(T)] = val_in[len(T)]; return v
        m = len(E_sets)
        E_bool = np.zeros((m, d), bool)
        for a, S in enumerate(E_sets):
            E_bool[a, list(S)] = True
        if m:
            ham = (E_bool[:, None, :] ^ E_bool[None, :, :]).sum(-1)
            Minv = np.linalg.inv(beta ** ham + 1e-6 * np.eye(m))
            W = np.stack([Ak(S) for S in E_sets], axis=1)
            G = G_prior - W @ Minv @ W.T
        else:
            Minv = np.zeros((0, 0)); W = np.zeros((d, 0)); G = G_prior
        Ginv = np.linalg.pinv(0.5 * (G + G.T), rcond=1e-11, hermitian=True)
        N = 1 << d
        for S in range(N):
            T = frozenset(int(j) for j in range(d) if Mem[S, j])
            if m:
                kmT = np.array([beta ** len(T ^ Sa) for Sa in E_sets])
                sol = Minv @ kmT
                v = 1 + 1e-6 - kmT @ sol
                c = Ak(T) - W @ sol
            else:
                v = 1 + 1e-6
                c = Ak(T)
            r = c @ Ginv @ c / v
            worst = max(worst, abs(r - bs[S]))
    print(f"  max |rho2_efficient - rho2_brute| over all 2^d coalitions x 3 designs: {worst:.2e}")


if __name__ == "__main__":
    validate(8, 0.6)
    validate(10, 0.3)
    validate(10, 0.9)
    for beta in (0.3, 0.6, 0.9):
        greedy_run(30, beta, steps=40)
    greedy_run(100, 0.6, steps=30)
