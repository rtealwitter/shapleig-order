# The ShaplEIG greedy has a closed form, and it barely moves

*Notes on the value-independent structure of the ShaplEIG selection rule
[RFMBF, ICML '26], with a replication on the paper's own games and code, and a
hybrid that keeps the accuracy at a fraction of the fitting cost. Notation
follows the paper: $p$ players, value function $\nu$, Shapley values
$\boldsymbol{\phi} = \mathbf{A}\boldsymbol{\nu}$, Hamming kernel
$k_\xi(S,T) = \prod_{j \in S \triangle T} \beta_j$ with
$\beta_j = \exp(-1/\ell_j^2)$.*

## 1. The selection sequence is a function of the lengthscales alone

The paper already notes the key fact (Section 3.3, "Adaptivity"): Gaussian
conditioning changes the posterior covariance in a way that depends only on
*which* coalitions were evaluated, never on the observed values, and the EIG
reads only the posterior covariance. With the lengthscales frozen, the entire
"adaptive" design is therefore determined before the first evaluation. The
observations steer the design through exactly one channel: the hyperparameter
refits. Two questions follow. What *is* the frozen-kernel sequence? And how
much of ShaplEIG's advantage survives when you replace the adaptive loop with
that fixed sequence?

## 2. Closed form for the frozen-kernel greedy

Take equal lengthscales, $\beta_j = \beta$ (the symmetric state before any
data). Everything diagonalizes in the parity basis
$\chi_U(S) = (-1)^{|U \cap S|}$:

- The Hamming kernel is the tensor product over players of
  $\bigl(\begin{smallmatrix}1 & \beta\\ \beta & 1\end{smallmatrix}\bigr)$, so
  its eigenvectors are the parities, with eigenvalues
  $\lambda_U = (1-\beta)^{|U|}(1+\beta)^{p-|U|}$. The prior mass decays
  geometrically in the parity degree $|U|$.
- The Shapley functional of a parity has a closed form. Writing $u = |U|$ and
  using $\frac{1}{p\binom{p-1}{s}} = \int_0^1 t^s(1-t)^{p-1-s}\,dt$, the
  marginal-contribution sum telescopes to
  $\phi_i(\chi_U) = -2\int_0^1 (1-2t)^{u-1}dt = -2/u$ for $i \in U$ with $u$
  odd, and $0$ otherwise. Even parities are invisible to Shapley values; this
  is the even-odd decomposition in eigenbasis form.

With prior and functional both explicit, every quantity in the EIG
(the vector $\mathbf{A}K_\xi(\mathbf{Z}, \mathbf{z})$, the matrix
$\mathbf{A}K_\xi(\mathbf{Z},\mathbf{Z})\mathbf{A}^\top$, all posterior
corrections) is computable without touching the $2^p$ lattice, either through
the paper's ESP machinery or through a Walsh-Hadamard transform. One more
ingredient makes the *argmax* cheap as well: any permutation of players that
fixes the evaluated design (as a set of coalitions) leaves every EIG score
invariant, so candidates only need to be scored once per orbit. After $m$
symmetric steps there are $O(p^2)$ orbits, not $2^p$ candidates, and the exact
greedy runs in polynomial time. (Numerical note: build the leave-one-player-out
generating polynomials as prefix and suffix products; synthetic division
amplifies rounding error by $\beta^{-p}$.)

Running this exact greedy gives the same sequence at every $p$ and every
$\beta$ we tested ($p$ up to $100$, $\beta \in \{0.3, 0.6, 0.9\}$, verified
against dense brute force to $10^{-10}$ for $p \le 10$):

1. $\emptyset$, then the grand coalition. They tie at step one, and whichever
   is taken, the other follows with $\rho^2 \approx 1$: by efficiency,
   $\sum_i \phi_i = \nu([p]) - \nu(\emptyset)$, so after seeing
   $\nu(\emptyset)$ the Shapley vector determines the second value up to
   noise.
2. Singleton and co-singleton pairs: $\{j\}$, then $[p]\setminus\{j\}$,
   through all players. Which player goes next is an exact tie; the sequence
   is unique only up to relabeling.
3. Once the $2p+2$ extremes are exhausted, complement pairs of half-size
   coalitions with balanced pairwise intersections. (For very smooth kernels,
   $\beta \to 1$, the middle phase starts earlier.)

Every step is followed by its complement. The reason is the even-odd split:
$\nu(T^c) = \nu(T) - 2\nu_{\mathrm{odd}}(T)$, the Shapley vector reads only
the odd part, and the prior concentrates on degree-one parities, so once
$\nu(T)$ is known, the Shapley vector explains almost all remaining
uncertainty in $\nu(T^c)$. Extremes-first mirrors the Shapley weights
$1/\binom{p-1}{\cdot}$, which peak at sizes $0, 1, p{-}1, p$; this is the same
weight concentration that Kernel SHAP and Leverage SHAP [MW, ICLR '25] exploit.
In short, the frozen-kernel EIG greedy rediscovers paired sampling at extreme
sizes [CL, NeurIPS '21], with a principled continuation into balanced middles.

## 3. How the greedy moves in distribution

The interesting comparison is against the *fully adaptive* ShaplEIG, refit
every iteration. Logging the selected coalition at each step of the paper's
own pipeline and plotting $\min(|T|, p-|T|)$, the distance of the selection to
the nearest endpoint of the size range, shows the adaptive rule's selection
distribution drifting exactly along the frozen-kernel schedule (right panels
of the attached figures, mean over 30 seeds with a 20-80% band):

- For roughly the first hundred iterations at $p = 10$, ShaplEIG selects at
  distance one to two from the endpoints: near-singletons and
  near-co-singletons.
- It then migrates to mid-size coalitions (distance three to five) for the
  remainder of the run, exactly the extremes-then-middles progression the
  closed form predicts.

What the refits actually contribute is the *timing* of that migration and
mild game-specific reordering, not the shape of the design. GP + Random, for
contrast, sits at the typical random distance of about $3.8$ forever.

## 4. Replication on the paper's games, plus two new arms

Setup: the authors' released code and precomputed dataset-valuation games
(Bike Sharing RF and GB, California Housing GB, $p = 10$, all 30 seeds),
with the paper's defaults: leverage-sampled initial design of size $p+1$ with
pairing, MAP lengthscale fitting with the BoTorch LogNormal prior, noise
$10^{-6}$, exhaustive EIG, 512 iterations, MSE against the exact Shapley
values, mean with SEM. Two arms are added to the paper's Figure 3 ablation:

- **Paired extremes (fixed order):** the frozen-kernel schedule above,
  replayed with no selection adaptivity, inside the otherwise identical
  pipeline (same init, same per-iteration refits, same posterior-mean
  estimate).
- **Hybrid:** select the remaining extremes in the fixed order first (their
  value does not depend on the lengthscales, so no fitting is needed to
  justify them), then switch to EIG selection; refit the lengthscales only at
  iterations $0, 1, 2, 4, 8, \dots$, ten fits per 512 iterations instead of
  512. Between refits the EIG updates incrementally under frozen
  hyperparameters, which is exactly the paper's own scalability mode. The
  geometric spacing matters: a constant interval of 32 leaves the early
  *estimates* on stale lengthscales and costs three orders of magnitude of
  MSE at $m \approx 32$, while dense-early refitting is cheap because the
  early fits are on few points.

Results (MSE, mean over 30 seeds; evaluations include the 11 initial ones):

**DV; RF; Bike Sharing** (cumulative fit + selection compute in parentheses)

| evals | ShaplEIG | Hybrid | Paired extremes | GP + Leverage | GP + Random |
|---|---|---|---|---|---|
| 32  | 1.9e-3 (10s) | 3.5e-5 (6s) | **1.0e-5** (5s) | 8.4e-3 | 1.3e-2 |
| 64  | 1.4e-5 (24s) | 1.6e-5 (13s) | **9.1e-6** (12s) | 3.8e-3 | 8.9e-3 |
| 128 | 1.4e-6 (57s) | **1.2e-6** (28s) | 7.1e-6 (32s) | 5.6e-6 | 5.5e-3 |
| 256 | **5.5e-7** (193s) | 5.8e-7 (59s) | 9.3e-7 (137s) | 2.7e-6 | 3.3e-3 |
| 523 | 6.0e-8 (1146s) | 6.9e-8 (**133s**) | 6.0e-8 (952s) | 1.1e-7 | 1.5e-3 |

**DV; GB; Bike Sharing**

| evals | ShaplEIG | Hybrid | Paired extremes | GP + Leverage | GP + Random |
|---|---|---|---|---|---|
| 32  | 2.3e-3 | 3.6e-5 | **3.1e-6** | 9.5e-3 | 1.4e-2 |
| 64  | 4.6e-6 | 5.4e-6 | **2.9e-6** | 4.5e-3 | 9.9e-3 |
| 128 | 7.1e-7 | **6.9e-7** | 2.5e-6 | 2.1e-6 | 6.5e-3 |
| 256 | 6.8e-7 | 6.3e-7 | **5.2e-7** | 1.2e-6 | 4.0e-3 |
| 523 | 5.7e-8 (1212s) | **5.6e-8** (**133s**) | 6.9e-8 (1216s) | 1.1e-7 | 2.1e-3 |

**DV; GB; California Housing**

| evals | ShaplEIG | Hybrid | Paired extremes | GP + Leverage | GP + Random |
|---|---|---|---|---|---|
| 32  | 1.6e-3 | 3.0e-5 | **1.2e-5** | 6.9e-3 | 1.0e-2 |
| 64  | 1.6e-5 | 1.8e-5 | **1.0e-5** | 3.2e-3 | 7.0e-3 |
| 128 | 2.8e-6 | **2.4e-6** | 8.1e-6 | 6.8e-6 | 4.4e-3 |
| 256 | 1.4e-6 | **1.3e-6** | 1.5e-6 | 3.5e-6 | 2.7e-3 |
| 523 | 1.5e-7 (1165s) | 1.8e-7 (**134s**) | **1.4e-7** (1237s) | 2.3e-7 | 1.3e-3 |

The paper's ablation replicates: ShaplEIG beats GP + Random by three to four
orders of magnitude, and GP + Random is revealed to be a weak non-adaptive
baseline rather than evidence that adaptivity is necessary. The fixed paired
schedule matches ShaplEIG at almost every budget and beats it by two to three
orders of magnitude around $m \approx 32$, where EIG under barely-fitted
lengthscales explores mid-size coalitions before the extremes are covered.
ShaplEIG's genuine adaptive edge is a factor of two to five in the
mid-budget range ($m \approx 100$ to $300$). The hybrid keeps that edge while
cutting the number of hyperparameter fits from 512 to 16; the time panels
show the resulting gap in cumulative compute.

## 5. Making the lengthscales cheap

The fitting, not the EIG, dominates ShaplEIG's cost (the paper reports up to
25 minutes per refit at $p \approx 100$). Three compounding reductions:

1. **Selection needs no lengthscales for the first $2p+2$ evaluations.** The
   extremes phase is justified under any $\beta$, so all fitting during it
   can be skipped or deferred.
2. **Refit on a sparse schedule.** EIG selection between refits is
   value-independent given $\beta$ and updates incrementally (the paper's
   $p > 16$ mode). Every 32nd iteration loses nothing measurable here; a
   geometric schedule (refit at $m = p{+}1, 2p, 4p, \dots$) is the natural
   general choice.
3. **Fit the odd part only.** Complement-paired designs let you repackage the
   $m$ observations as $m/2$ observations of the odd part
   $\nu_{\mathrm{odd}}$ under the odd kernel
   $k_{\mathrm{odd}} = \tfrac12(\prod_{j \in S \triangle T}\beta_j -
   \prod_{j \notin S \triangle T}\beta_j)$, with no change to the Shapley
   posterior. Every Cholesky inside the marginal-likelihood optimization
   shrinks by $2\times$ per side, an $8\times$ saving per fit, and the
   marginal likelihood stops being distracted by the even part of the game,
   which the Shapley values never see.

Together: the same estimates, with $O(\log m)$ fits that are each $8\times$
cheaper, and a selection rule that is a lookup for its first $2p+2$ steps.
