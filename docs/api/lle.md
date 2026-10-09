# Locally linear embedding

The LLE family (LLE, modified LLE, Hessian LLE, LTSA) describes every point by its $k$ nearest neighbours and looks
for low-dimensional coordinates that keep those local descriptions. All of
them end in the same eigenproblem: one positive semidefinite block per point
summed into a sparse $N \times N$ matrix $M$, whose bottom eigenvectors
(after the constant one) are the embedding.

```python
import jax
import manipy

roll, t = manipy.datasets.swiss_roll(2000, key=jax.random.key(0))
lle = manipy.LocallyLinearEmbedding(n_components=2, n_neighbors=12)
Y = lle.fit(roll).embedding
Y_mlle = (
    manipy.LocallyLinearEmbedding(
        n_neighbors=12, method="modified", eigen_solver="arpack"
    )
    .fit(roll)
    .embedding
)
Y_ltsa = (
    manipy.LocallyLinearEmbedding(n_neighbors=12, method="ltsa").fit(roll).embedding
)
```

The neighbours come from kernellib's `nearest_neighbors`; the per-point
problems are `vmap`-ped; $M$ is a `gaussx.SparseOperator`.

## Formulation

Let $\mathcal N_i$ be the $k$ nearest neighbours of $x_i$ (itself excluded),
$Z_i \in \mathbb R^{k \times D}$ their offsets $x_j - x_i$, and
$C_i = Z_i Z_i^\top$ the local Gram matrix. With $S_i$ selecting the rows
$(i, \mathcal N_i)$ (LLE, MLLE) or $\mathcal N_i$ (HLLE, LTSA),

$$
M = \sum_i S_i^\top G_i S_i,
$$

and the embedding is the eigenvectors $u_2, \dots, u_{n+1}$ of the $2$-nd to
$(n+1)$-th smallest eigenvalues of $M$ (the smallest, $0$, belongs to
$\mathbf 1$, an exact null vector of every $G_i$). Precisely: the $n+1$
smallest eigenvectors $U$ are computed, $\mathbf 1$ is projected out of
their span, and $M$ is re-solved on the remaining $n$ directions
(Rayleigh–Ritz). When the null space is degenerate (a flat sheet under HLLE
or LTSA has three null vectors), `eigh` may return any basis of it, and
dropping its first column would keep a mixture with $\mathbf 1$; the
projection does not. Otherwise it changes nothing. They are orthonormal, so $Y^\top Y = I$ and $\mathbf 1^\top Y =
0$. `reconstruction_error` is $\sum_{k=2}^{n+1}\lambda_k$, scikit-learn's
`reconstruction_error_`. Each column is signed so its largest-magnitude
entry is positive.

### LLE

Roweis & Saul (2000). The reconstruction weights solve
$\min_{w}\|x_i - \sum_{j\in\mathcal N_i} w_j x_j\|^2$ with
$\mathbf 1^\top w = 1$; with the Lagrangian and a ridge,

$$
(C_i + r_i I)\, \tilde w_i = \mathbf 1,\qquad
w_i = \tilde w_i / \mathbf 1^\top \tilde w_i,\qquad
r_i = \texttt{reg} \cdot \operatorname{tr} C_i
$$

(`reg` itself if the trace is zero). The ridge makes the solve well posed
when $k > D$. With $W$ the $N \times N$ weight matrix,
$M = (I - W)^\top (I - W) = \sum_i v_i v_i^\top$, $v_i = e_i - W^\top e_i$,
which is the block $G_i = v v^\top$, $v = (1, -w_i)$ on $(i, \mathcal N_i)$.

### Modified LLE

Zhang & Wang (2007), as scikit-learn's `method="modified"`. With
$C_i = V_i \operatorname{diag}(\sigma_1 \ge \dots \ge \sigma_k) V_i^\top$
($\sigma_j = 0$ beyond the $\min(D, k)$ non-zero ones):

1. regularised weights $w_i \propto V_i (\Sigma + r_i I)^{-1} V_i^\top \mathbf 1$,
   $r_i = 10^{-3} \sum_j \sigma_j$, normalised to sum to one;
2. $\eta = \operatorname{median}_i\, \sum_{j > n}\sigma_j / \sum_{j \le n}\sigma_j$;
3. $s_i$, the size of the "almost null space": the number of trailing
   eigenvalues whose share $\sum_{j > k - s}\sigma_j / \sum_{j \le k - s}\sigma_j$
   stays below $\eta$, plus the $k - \min(D, k)$ zero ones;
4. with $\bar V_i$ the bottom $s_i$ eigenvectors,
   $\alpha_i = \|\bar V_i^\top \mathbf 1\| / \sqrt{s_i}$ and the Householder
   vector $h_i \propto \alpha_i \mathbf 1 - \bar V_i^\top \mathbf 1$ (zero if
   its norm is below `modified_tol`),
   $W_i = \bar V_i (I - 2 h_i h_i^\top) + (1 - \alpha_i)\, w_i \mathbf 1^\top$;
5. $G_i = \hat W_i \hat W_i^\top$ with $\hat W_i = [-\mathbf 1^\top; W_i]$
   on $(i, \mathcal N_i)$.

The $s_i$ vary by point; to `vmap`, $\bar V_i$ is $V_i$ with all but the last
$s_i$ columns masked to zero, which leaves every product unchanged.

### Hessian LLE

Donoho & Grimes (2003). With $U_i \in \mathbb R^{k \times n}$ the top $n$
left singular vectors of the *centred* neighbourhood (local tangent
coordinates) and $d_p = n(n+1)/2$,

$$
Y_i = \big[\mathbf 1,\; U_i,\; (u_a \odot u_b)_{1 \le a \le b \le n}\big]
\in \mathbb R^{k \times (1 + n + d_p)},\qquad Y_i = Q_i R_i
$$

(reduced QR), and $H_i = Q_i[:, n+1:]$, the $d_p$ orthonormal columns of the
quadratic part: the local Hessian estimator. $G_i = H_i H_i^\top$ on
$\mathcal N_i$, so $M$ is the discrete Hessian quadratic form
$\sum_i \|H_i^\top f_{\mathcal N_i}\|^2$, whose null space holds the
constant and the $n$ isometric coordinates. It needs $k > n(n+3)/2$.

**scikit-learn differs.** Its `method="hessian"` takes the *full* QR of
$Y_i$, so $H_i$ spans the whole complement of $[\mathbf 1, U_i]$ and $G_i$
equals LTSA's block: its embedding and reconstruction error are LTSA's (to
$10^{-12}$; `test_sklearns_hessian_is_ltsa`). This page follows Donoho &
Grimes. Like scikit-learn it divides by no column sums (they vanish, since
$H_i \perp \mathbf 1$) and skips the paper's final $R^{-1/2}$
renormalisation: the embedding is the orthonormal null-space basis.

### LTSA

Zhang & Zha (2004). With $U_i$ as above, $G_i = I_k - \bar G_i \bar G_i^\top$,
$\bar G_i = [\mathbf 1/\sqrt k,\, U_i]$ on $\mathcal N_i$: the projection
off each neighbourhood's affine tangent space, so $M$ penalises the part of
the global coordinates that the local tangent coordinates cannot explain.

### Eigensolvers

| `eigen_solver` | Method | Notes |
|---|---|---|
| `"dense"` (default) | `jnp.linalg.eigh` of the materialised $M$ | Exact, $O(N^3)$ |
| `"arpack"` | SciPy `eigsh`, shift-invert about $\sigma = -10^{-10}c$ | Sparse, CPU, not traced; residual-checked |

$c$ is the Gershgorin bound on $\lambda_{\max}(M)$. The shift makes
$M - \sigma I$ positive definite, so its factorisation is safe although $M$
is singular. Every ARPACK pair must satisfy
$\|Mu - \lambda u\| \le \sqrt{\varepsilon}\, c$, or `fit` raises.

There is no Lanczos option. The bottom of an LLE spectrum is clustered: on
a 400-point swiss roll $c \approx 6$ and the wanted eigenvalues are
$10^{-7}$–$10^{-6}$. Lanczos on the flipped $cI - M$ (`gaussx.eig` with
`rank=r`) leaves residuals of $10^{-3}$ at $r = 100$ and $10^{-5}$ at
$r = 300$, and is only accurate at $r \approx N$, the regime where gaussx's
Lanczos is unreliable
([gaussx#649](https://github.com/jejjohnson/gaussx/issues/649)).
Shift-invert resolves the clustered bottom directly.

## Pseudocode

```text
fit(X):
    N_i     ← k nearest neighbours of each x_i            # kernellib
    Z_i     ← X[N_i] − x_i;   C_i ← Z_i Z_iᵀ               # vmapped over i
    standard:  w_i ← solve(C_i + reg·tr(C_i)·I, 1);  w_i ← w_i / Σ w_i
               G_i ← v vᵀ,  v = (1, −w_i)
    modified:  (σ, V) ← eigh(C_i), descending;  w_i, η, s_i, α_i, h_i as above
               W_i ← V̄_i (I − 2 h hᵀ) + (1 − α_i) w_i 1ᵀ;   G_i ← Ŵ Ŵᵀ, Ŵ = [−1ᵀ; W_i]
    hessian:   U_i ← top-n eigenvectors of centred Gram;  Q ← qr([1, U_i, U_i⊙U_i])
               H_i ← Q[:, n+1:];   G_i ← H_i H_iᵀ                  (on N_i)
    ltsa:      Ḡ ← [1/√k, U_i];    G_i ← I − Ḡ Ḡᵀ                  (on N_i)
    M       ← SparseOperator.from_coo over the blocks (overlaps summed)
    U       ← n+1 smallest eigenvectors of M (dense eigh | ARPACK shift-invert)
    V       ← orthonormal basis of (I − 11ᵀ/N) span(U), dropping the 1 direction
    (λ, S)  ← eigh(Vᵀ M V);   Y ← V S
    return Y (signs fixed), λ, Σ λ
```

## References

- Roweis, S. T. & Saul, L. K. (2000). Nonlinear dimensionality reduction by
  locally linear embedding. *Science*, 290(5500), 2323–2326.
  [doi:10.1126/science.290.5500.2323](https://doi.org/10.1126/science.290.5500.2323)
- Saul, L. K. & Roweis, S. T. (2003). Think globally, fit locally:
  unsupervised learning of low dimensional manifolds. *Journal of Machine
  Learning Research*, 4, 119–155.
- Donoho, D. L. & Grimes, C. (2003). Hessian eigenmaps: locally linear
  embedding techniques for high-dimensional data. *Proceedings of the
  National Academy of Sciences*, 100(10), 5591–5596.
  [doi:10.1073/pnas.1031596100](https://doi.org/10.1073/pnas.1031596100)
- Zhang, Z. & Zha, H. (2004). Principal manifolds and nonlinear
  dimensionality reduction via tangent space alignment. *SIAM Journal on
  Scientific Computing*, 26(1), 313–338.
  [doi:10.1137/S1064827502419154](https://doi.org/10.1137/S1064827502419154)
- Zhang, Z. & Wang, J. (2007). MLLE: Modified locally linear embedding using
  multiple weights. *Advances in Neural Information Processing Systems 19*,
  1593–1600.

## API

::: manipy.LocallyLinearEmbedding
