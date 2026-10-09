# Out-of-sample extension

Spectral embeddings (Laplacian and Schrödinger eigenmaps, diffusion maps)
are transductive: they embed the points they were fitted on. The Nyström
extension (Bengio et al., 2004) evaluates the eigen-equation at a new point
instead, which gives a `transform` for points the model has never seen. At
a training point it gives back the fitted embedding exactly.

```python
import jax
import kernellib as kl
import manipy

roll, t = manipy.datasets.swiss_roll(3000, key=jax.random.key(0))
train, test = roll[:2000], roll[2000:]

le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=12).fit(train)  # from kernellib
Y_test = manipy.NystromExtension().fit(le, train).transform(test)

dm = manipy.DiffusionMaps(n_components=2, alpha=1.0, t=4, n_neighbors=12).fit(train)
Y_test_dm = manipy.NystromExtension().fit(dm, train).transform(test)
```

The eigenmaps are kernellib's: manipy extends them and does not re-export
them. In scikit-learn, `manipy.sklearn.NystromExtension(estimator)` wraps
`manipy.sklearn.DiffusionMaps` or kernellib's
`kernellib.sklearn.LaplacianEigenmaps` / `SchrodingerEigenmaps` into a
transformer.

## Formulation

All three models are eigenvectors of a random walk on an affinity $W$ with
degrees $d = W\mathbf 1$:

| Model | Eigen-equation | Random-walk eigenvalue $\mu_k$ | Stored $y_k$ |
|---|---|---|---|
| Laplacian eigenmaps | $L y = \lambda D y$, $L = D - W$ | $1 - \lambda_k$ | `embedding` |
| Schrödinger eigenmaps | $(L + \alpha V) y = \lambda D y$ | $1 - \lambda_k$ (where $V$'s row is zero) | `embedding` |
| Diffusion maps | $D_d^{-1} K^{(\alpha)} \psi = \lambda \psi$ | $\lambda_k$ | `embedding` $= \lambda_k^t\psi_k$ |

In each case $\sum_j \frac{W_{ij}}{d_i} y_{jk} = \mu_k\, y_{ik}$. Replacing
row $i$ by the affinities of a new point $x$ defines the extension

$$
y_k(x) = \frac{1}{\mu_k}\sum_j \frac{w^{(\alpha)}(x, x_j)}{\sum_l w^{(\alpha)}(x, x_l)}\; y_{jk},
\qquad
w^{(\alpha)}(x, x_j) = \frac{w(x, x_j)}{q(x)^\alpha\, q_j^\alpha},
$$

with $q(x) = \sum_j w(x, x_j)$, $q_j$ the training degrees and $\alpha$ the
diffusion map's anisotropy ($\alpha = 0$ for eigenmaps). For eigenmaps this
is Bengio et al.'s $y_k(x) = \frac{1}{1-\lambda_k}\sum_j
\frac{w(x,x_j)}{d(x)} y_{jk}$.

**The affinity of a new point** is the one the training graph would have
given it:

- **Dense kernel** (diffusion maps with `n_neighbors=None`):
  $w(x, x_j) = k(x, x_j)$ for every $j$, including $j$ with $x_j = x$ (the
  Gram matrix has self-loops).
- **Symmetrised k-NN graph** (eigenmaps, and diffusion maps with
  `n_neighbors=k`). The training graph joins $i$ and $j$ if either lists the
  other among its $k$ nearest. So $x$ is joined to $x_j$ if $x_j$ is among
  the $k$ nearest training points of $x$, **or** $\|x - x_j\| \le r_j$,
  $x_j$'s distance to its own $k$-th neighbour (then $x_j$ would list $x$).
  The edge is weighted by the graph's kernel: the heat kernel at the
  model's bandwidth (kernellib's default is the median k-NN distance), or 1
  for connectivity weights. A training point at distance zero is the query
  itself and gets no edge, because the graph has no self-loops.

At a training point $x_i$ both rules reproduce row $i$ of $W$ exactly, so
$y(x_i) = y_i$. The tests check this for every model and every $\alpha$,
and check that the affinity rows equal kernellib's `adjacency_matrix`.

**Limits.**

- Schrödinger eigenmaps: a new point carries no potential, so the formula is
  exact at training points whose potential row is zero (the unlabelled
  points of a label or barrier potential).
- Diffusion maps on a graph: bridge edges added by `ensure_connected` are
  not reproduced.
- Eigenmaps fitted on a precomputed graph, or with
  `constraint="identity"` (no random walk), are rejected.
- On a k-NN graph the extension is only piecewise smooth, since a small move
  can change the neighbour set. A dense kernel makes it smooth.

Ties at exactly $r_j$ get a relative slack of $100\,\varepsilon_{\text{mach}}$,
and a query within that rounding of a training point counts as coincident
with it.

## Pseudocode

```text
fit(model, X):
    Y, μ  ← model.embedding, (1 − λ) for eigenmaps | λ for diffusion maps
    w     ← heat kernel at the model's bandwidth | connectivity | DM's fitted kernel
    k, r  ← the graph's k and each x_j's k-th neighbour distance (graph models)
    C     ← Y / μ;   keep α and q_j (diffusion maps)

transform(X_new):                                   # O(M N)
    dense:  W ← w(X_new, X)
    graph:  D ← distances, 0 → "self" (excluded)
            E ← D ≤ (k-th smallest of each row)  or  D ≤ r_j
            W ← w(X_new, X) on E, else 0
    if α > 0:  W ← W / q(x)^α / q_j^α,   q(x) = W 1
    return (W / W 1) C
```

## References

- Bengio, Y., Paiement, J.-F., Vincent, P., Delalleau, O., Le Roux, N. &
  Ouimet, M. (2004). Out-of-sample extensions for LLE, Isomap, MDS,
  Eigenmaps, and spectral clustering. *Advances in Neural Information
  Processing Systems 16*, 177–184.
- Williams, C. K. I. & Seeger, M. (2001). Using the Nyström method to speed
  up kernel machines. *Advances in Neural Information Processing Systems
  13*, 682–688.
- Coifman, R. R. & Lafon, S. (2006). Geometric harmonics: a novel tool for
  multiscale out-of-sample extension of empirical functions. *Applied and
  Computational Harmonic Analysis*, 21(1), 31–52.
  [doi:10.1016/j.acha.2005.07.005](https://doi.org/10.1016/j.acha.2005.07.005)

## API

::: manipy.NystromExtension
