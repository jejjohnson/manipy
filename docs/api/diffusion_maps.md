# Diffusion maps

Diffusion maps (Coifman & Lafon, 2006) embed points by the leading
eigenfunctions of a random walk on the data. Euclidean distance in the
embedding is the *diffusion distance*: two points are close when the walk
started from either spreads out in the same way after $t$ steps. The
anisotropy $\alpha$ controls how much the sampling density shapes the walk.
With $\alpha = 1$, the walk approximates heat flow on the manifold itself,
whatever the density.

```python
import jax
import manipy

roll, t = manipy.datasets.swiss_roll(2000, key=jax.random.key(0))
dm = manipy.DiffusionMaps(n_components=2, alpha=1.0, t=4, n_neighbors=12).fit(roll)
dm.embedding  # λ_k^t ψ_k, k = 1, 2
```

Diffusion maps live in manipy, not kernellib: they are a dimensionality
reduction method, not a kernel or graph primitive. The graph and kernels are
kernellib's (`knn_graph`, `RBF`, `estimate_lengthscale`), and the Lanczos
solver is gaussx's `eig`.

## Formulation

**Affinity.** With `n_neighbors=None`, $K$ is the dense Gram matrix
$K_{ij} = k(x_i, x_j)$, self-loops included. With `n_neighbors=k`, it is
$k(x_i, x_j)$ on the edges of the symmetrised k-NN graph, joined into one
component (`ensure_connected=True`), with no self-loops. The default kernel
is the heat kernel $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$, with $\sigma$
set by `bandwidth`, or by default the median distance over all pairs (Gram)
or over the k-NN distances (graph). Coifman & Lafon's
$\exp(-\|x-y\|^2/\varepsilon)$ has $\varepsilon = 2\sigma^2$. Any kernellib
kernel can be passed instead.

**Anisotropic normalisation.** With the kernel density estimate
$q = K\mathbf 1$,

$$
K^{(\alpha)} = D_q^{-\alpha} K D_q^{-\alpha},\qquad
d = K^{(\alpha)}\mathbf 1,\qquad
P = D_d^{-1} K^{(\alpha)}.
$$

$P$ is row-stochastic, with stationary distribution
$\pi = d / \sum_j d_j$. As $\varepsilon \to 0$, its generator
$(I - P)/\varepsilon$ tends to:

| $\alpha$ | Limit operator | Meaning |
|---|---|---|
| $0$ | $\Delta - 2\nabla U \cdot \nabla$ (normalised graph Laplacian) | Density distorts the geometry |
| $\tfrac12$ | Fokker–Planck, $\Delta - \nabla U \cdot \nabla$ | Langevin dynamics in the potential $U = -\log p$ (Nadler et al., 2006) |
| $1$ | Laplace–Beltrami $\Delta$ | Geometry only: the density is removed |

up to constants, with $p = e^{-U}$ the sampling density.

**Eigenpairs through the symmetric conjugate.** $P$ is similar to
$A = D_d^{-1/2} K^{(\alpha)} D_d^{-1/2} = D_d^{1/2} P D_d^{-1/2}$, which is
symmetric (dense, or a `gaussx.SparseOperator` on the graph). If
$A\phi_k = \lambda_k\phi_k$ with $1 = \lambda_0 > \lambda_1 \ge \dots$, the
right eigenvectors of $P$ are

$$
\psi_k = \sqrt{\textstyle\sum_j d_j}\; D_d^{-1/2}\phi_k,
\qquad
\sum_i \pi_i\, \psi_k(i)\, \psi_l(i) = \delta_{kl},
$$

so $\psi_0 \equiv 1$, which is dropped.

**Diffusion map.** At time $t$,

$$
\Psi_t(x_i) = \big(\lambda_1^t \psi_1(i), \dots, \lambda_n^t \psi_n(i)\big),
$$

stored in `embedding`, with $\lambda$ in `eigenvalues`, $\psi$ in
`eigenvectors` and $q$ in `degrees`. With all $N - 1$ components,

$$
\|\Psi_t(x_i) - \Psi_t(x_j)\|^2 = D_t^2(i, j)
= \sum_z \frac{\big(P^t_{iz} - P^t_{jz}\big)^2}{\pi_z},
$$

the diffusion distance (checked exactly in the tests). Larger $t$ shrinks
the fast-decaying coordinates, which makes the embedding coarser.

**Eigensolvers.** `"dense"`: `eigh` of $A$, exact. `"lanczos"`:
`gaussx.eig(A, rank=r)` with $r = \min(N/2, \max(20(n+1), 100))$, well below
$N$, because gaussx's Lanczos is unreliable when the rank is close to $N$
([gaussx#649](https://github.com/jejjohnson/gaussx/issues/649)). Every pair
must satisfy $\|A\phi - \lambda\phi\| \le \sqrt{\varepsilon_{\text{mach}}}$
(the spectrum of $A$ lies in $[-1, 1]$). A failing run is repeated with
twice the rank, up to $N/2$, and then `fit` raises. The scikit-learn
adapter solves inputs of up to 200 samples densely.

## Pseudocode

```text
fit(X):
    k      ← kernel, or RBF(σ) with σ = bandwidth | median distance
    K      ← k(X, X)                                  (dense, self-loops)
           | k on the edges of knn_graph(X, n_neighbors, ensure_connected)
    q      ← K 1;   K_α ← q^{−α} ∘ K ∘ q^{−α};   d ← K_α 1
    A      ← d^{−1/2} ∘ K_α ∘ d^{−1/2}               (symmetric)
    (λ, φ) ← top n+1 eigenpairs of A (eigh | gaussx.eig, rank-doubling, residual check)
    drop (λ_0 = 1, φ_0 ∝ √d)
    ψ      ← √(Σ d) · φ / √d                         (π-orthonormal, signs fixed)
    return Ψ_t = ψ · λ^t,  λ,  ψ,  q
```

## References

- Coifman, R. R. & Lafon, S. (2006). Diffusion maps. *Applied and
  Computational Harmonic Analysis*, 21(1), 5–30.
  [doi:10.1016/j.acha.2006.04.006](https://doi.org/10.1016/j.acha.2006.04.006)
- Coifman, R. R., Lafon, S., Lee, A. B., Maggioni, M., Nadler, B., Warner, F.
  & Zucker, S. W. (2005). Geometric diffusions as a tool for harmonic
  analysis and structure definition of data: diffusion maps. *Proceedings of
  the National Academy of Sciences*, 102(21), 7426–7431.
  [doi:10.1073/pnas.0500334102](https://doi.org/10.1073/pnas.0500334102)
- Nadler, B., Lafon, S., Coifman, R. R. & Kevrekidis, I. G. (2006).
  Diffusion maps, spectral clustering and reaction coordinates of dynamical
  systems. *Applied and Computational Harmonic Analysis*, 21(1), 113–127.
  [doi:10.1016/j.acha.2005.07.004](https://doi.org/10.1016/j.acha.2005.07.004)

## API

::: manipy.DiffusionMaps
