# Shared GP-surrogate machinery for the ShaplEIG baselines benchmark.
# Hamming-kernel GP over the coalition lattice with per-player beta_j.
# The Walsh-Hadamard identity K = 2^-d H diag(D) H (H_{S,U} = (-1)^{|S&U|},
# D_U = prod_{j in U}(1-b_j) prod_{j notin U}(1+b_j)) lets us form A K, A K A^T,
# and kernel columns in O(2^d d^2) without materializing the 2^d x 2^d kernel.
import numpy as np
from math import comb
from scipy.optimize import minimize


def membership(d):
    N = 1 << d
    return ((np.arange(N)[:, None] >> np.arange(d)) & 1).astype(bool)


def shapley_A(d, Mem=None):
    Mem = membership(d) if Mem is None else Mem
    N = Mem.shape[0]
    sizes = Mem.sum(1)
    win = np.array([1.0 / (d * comb(d - 1, s - 1)) if s >= 1 else 0.0 for s in range(d + 1)])
    wout = np.array([1.0 / (d * comb(d - 1, s)) if s <= d - 1 else 0.0 for s in range(d + 1)])
    A = np.where(Mem.T, win[sizes][None, :], -wout[sizes][None, :])
    return A


def AH_matrix(d, Mem=None):
    # [A H]_{i,U} = phi_i(chi_U) = -2/|U| if i in U and |U| odd, else 0
    Mem = membership(d) if Mem is None else Mem
    usz = Mem.sum(1)
    odd = (usz % 2 == 1)
    return np.where(Mem.T & odd[None, :], -2.0 / np.maximum(usz, 1)[None, :], 0.0)


def fwht(a):
    # Walsh transform over the last axis: out[S] = sum_U (-1)^{|S & U|} a[U]
    a = np.array(a, dtype=float, copy=True)
    shp = a.shape
    n = shp[-1]
    a = a.reshape(-1, n)
    h = 1
    while h < n:
        b = a.reshape(-1, n // (2 * h), 2, h)
        x = b[:, :, 0, :].copy()
        y = b[:, :, 1, :].copy()
        b[:, :, 0, :] = x + y
        b[:, :, 1, :] = x - y
        h *= 2
    return a.reshape(shp)


def eigen_D(betas, Mem):
    # D_U for all U, via log-accumulation over players
    lo = np.log1p(-betas)   # log(1 - b_j)
    hi = np.log1p(betas)    # log(1 + b_j)
    return np.exp(Mem @ (lo - hi) + hi.sum())


def kernel_mm(betas, E_bool):
    # K on the evaluated coalitions: prod_j beta_j^{xor}
    X = E_bool[:, None, :] ^ E_bool[None, :, :]
    return np.exp(X @ np.log(betas))


def kernel_cols(betas, Mem, E_bool):
    # K[:, E] (N x m) directly: prod over players of beta_j^{xor}
    logb = np.log(betas)
    L = np.zeros((Mem.shape[0], E_bool.shape[0]))
    for j in range(Mem.shape[1]):
        L += logb[j] * (Mem[:, j][:, None] ^ E_bool[:, j][None, :])
    return np.exp(L)


# ---------------- marginal-likelihood fit ----------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll_and_grad(params, E_bool, y, Xstack):
    d = E_bool.shape[1]
    z, w = params[:d], params[d]
    betas = np.clip(_sigmoid(z), 1e-6, 1 - 1e-6)
    sig2 = np.exp(w)
    K = np.exp(Xstack.transpose(1, 2, 0) @ np.log(betas))
    m = K.shape[0]
    M = K + sig2 * np.eye(m)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return 1e10, np.zeros_like(params)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    nll = 0.5 * y @ alpha + np.log(np.diag(L)).sum()
    Minv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(m)))
    P = Minv - np.outer(alpha, alpha)
    R = P * K
    gz = np.empty(d)
    for j in range(d):
        gz[j] = 0.5 * (R * Xstack[j]).sum() * (1 - betas[j])
    gw = 0.5 * np.trace(P) * sig2
    return nll, np.concatenate([gz, [gw]])


def fit_gp(E_bool, y, z0=None, w0=None, maxiter=60, restarts=(0.0, 2.0)):
    """Standardizes y, fits (beta, sigma2) by marginal likelihood.
    Returns betas, sig2, y_center, y_scale."""
    d = E_bool.shape[1]
    mu, sc = y.mean(), y.std()
    sc = sc if sc > 1e-12 else 1.0
    ys = (y - mu) / sc
    Xstack = (E_bool[:, None, :] ^ E_bool[None, :, :]).transpose(2, 0, 1).astype(float)
    best = None
    starts = [(np.full(d, s), np.log(1e-2)) for s in restarts]
    if z0 is not None:
        starts = [(z0, w0 if w0 is not None else np.log(1e-2))]
    for zs, ws in starts:
        res = minimize(nll_and_grad, np.concatenate([zs, [ws]]),
                       args=(E_bool, ys, Xstack), jac=True, method="L-BFGS-B",
                       bounds=[(-8, 8)] * d + [(np.log(1e-6), np.log(1.0))],
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    z, w = best.x[:d], best.x[d]
    return np.clip(_sigmoid(z), 1e-6, 1 - 1e-6), np.exp(w), mu, sc, z, w


def shapley_estimate(betas, sig2, Mem, AHm, E_bool, y, y_center, y_scale):
    """phi_hat = A K_{.m} (K_mm + sig2 I)^{-1} y_std * scale."""
    d = Mem.shape[1]
    N = Mem.shape[0]
    D = eigen_D(betas, Mem)
    AK = fwht(AHm * D[None, :]) / N            # d x N: A K
    E_idx = (E_bool @ (1 << np.arange(d))).astype(int)
    W = AK[:, E_idx]                           # A K_{.m}
    M = kernel_mm(betas, E_bool) + sig2 * np.eye(E_bool.shape[0])
    ys = (y - y_center) / y_scale
    return (W @ np.linalg.solve(M, ys)) * y_scale


def eig_scores(betas, sig2, Mem, AHm, E_bool):
    """rho^2 for every coalition given the evaluated set; O(N(m^2+d^2))."""
    N, d = Mem.shape
    D = eigen_D(betas, Mem)
    AHD = AHm * D[None, :]
    AK = fwht(AHD) / N                      # d x N: A K
    G_prior = AHD @ AHm.T / N               # A K A^T
    m = E_bool.shape[0]
    if m:
        Kc = kernel_cols(betas, Mem, E_bool)      # N x m
        M = kernel_mm(betas, E_bool) + sig2 * np.eye(m)
        Minv = np.linalg.inv(M)
        E_idx = (E_bool @ (1 << np.arange(d))).astype(int)
        W = AK[:, E_idx]                          # A K_{.m}
        Sol = Kc @ Minv                           # N x m
        v = 1.0 + sig2 - np.einsum('nm,nm->n', Sol, Kc)
        C = AK - W @ Sol.T                        # d x N: c_T columns
        G = G_prior - W @ Minv @ W.T
    else:
        v = np.full(N, 1.0 + sig2)
        C = AK
        G = G_prior
        E_idx = np.array([], int)
    Ginv = np.linalg.pinv(0.5 * (G + G.T), rcond=1e-11, hermitian=True)
    num = np.einsum('iT,ij,jT->T', C, Ginv, C)
    rho2 = num / np.maximum(v, 1e-15)
    rho2[E_idx] = -np.inf
    return rho2


def fixed_schedule(d, m_max, beta=0.5, sig2=1e-4):
    """Value-independent greedy design with a frozen symmetric kernel."""
    Mem = membership(d)
    AHm = AH_matrix(d, Mem)
    betas = np.full(d, beta)
    E_bool = np.zeros((0, d), bool)
    masks = []
    for _ in range(m_max):
        r = eig_scores(betas, sig2, Mem, AHm, E_bool)
        pick = int(np.argmax(r))
        row = Mem[pick]
        masks.append(pick)
        E_bool = np.vstack([E_bool, row[None, :]])
    return np.array(masks, dtype=np.int64)


def self_test():
    rng = np.random.default_rng(0)
    d = 6
    Mem = membership(d)
    N = 1 << d
    betas = rng.uniform(0.2, 0.9, d)
    # dense kernel
    K = np.ones((N, N))
    for j in range(d):
        mj = Mem[:, j]
        K *= np.where(mj[:, None] ^ mj[None, :], betas[j], 1.0)
    H = np.where((Mem.astype(int) @ Mem.T.astype(int)) % 2 == 1, -1.0, 1.0)
    D = eigen_D(betas, Mem)
    assert np.allclose(K, (H * D[None, :]) @ H.T / N), "Walsh kernel identity"
    A = shapley_A(d, Mem)
    assert np.allclose(A @ H, AH_matrix(d, Mem)), "AH closed form"
    assert np.allclose(fwht(np.eye(N)[3]), H[3]), "fwht sign convention"
    assert np.allclose(A.sum(1), 0), "rows of A sum to zero"
    # eig_scores against dense computation
    E_bool = Mem[rng.choice(N, 5, replace=False)]
    sig2 = 1e-3
    Kc = kernel_cols(betas, Mem, E_bool)
    M = kernel_mm(betas, E_bool) + sig2 * np.eye(5)
    Sigma = K - Kc @ np.linalg.solve(M, Kc.T)
    G = A @ Sigma @ A.T
    Ginv = np.linalg.pinv(0.5 * (G + G.T), rcond=1e-11, hermitian=True)
    v = np.diag(Sigma) + sig2
    dense = np.einsum('iT,ij,jT->T', A @ Sigma, Ginv, A @ Sigma) / v
    fast = eig_scores(betas, sig2, Mem, AH_matrix(d, Mem), E_bool)
    keep = np.isfinite(fast)
    assert np.allclose(dense[keep], fast[keep], atol=1e-8), "eig_scores dense check"
    print("gp_core self-test: all passed")


if __name__ == "__main__":
    self_test()
