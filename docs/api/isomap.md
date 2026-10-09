# Isomap

Isomap embeds points so that Euclidean distances in the embedding match
*geodesic* distances along the data manifold, estimated as shortest paths
through a k-nearest-neighbour graph. On a swiss roll it unrolls the sheet
rather than flattening it onto itself.

```python
import jax
import manipy

roll, t = manipy.datasets.swiss_roll(2000, key=jax.random.key(0))
Y = manipy.Isomap(n_components=2, n_neighbors=12).fit(roll).embedding
Y_fast = (
    manipy.Isomap(n_components=2, n_neighbors=12, n_landmarks=200).fit(roll).embedding
)
manipy.metrics.trustworthiness(roll, Y, n_neighbors=10)
```

The graph is kernellib's `knn_graph`. The shortest paths run on the CPU
through `scipy.sparse.csgraph.dijkstra`, so `fit` is eager: it is not traced
and cannot be `jit`-ted or differentiated.

## Formulation

**Geodesics.** $G$ is the k-NN graph of $X \in \mathbb R^{N \times D}$,
symmetrised (an edge if either point lists the other) and joined into one
component by the shortest bridging edges (`ensure_connected=True`, as
scikit-learn's `Isomap` does). Edge $(i, j)$ has length
$\|x_i - x_j\|$, and the geodesic distance $D_{ij}$ is the shortest-path
length between $i$ and $j$ in $G$.

**Classical MDS** (Torgerson, 1952). With $D^{(2)}$ the element-wise squares
and $J = I - \tfrac1N \mathbf 1 \mathbf 1^\top$,

$$
B = -\tfrac12 J D^{(2)} J, \qquad
B_{ij} = -\tfrac12\big(D^{(2)}_{ij} - \bar D^{(2)}_{i\cdot} - \bar D^{(2)}_{\cdot j}
+ \bar D^{(2)}_{\cdot\cdot}\big),
$$

the Gram matrix of centred points when $D$ is Euclidean. The embedding is
$Y = U_n \Lambda_n^{1/2}$, with $(\Lambda_n, U_n)$ the $n$ largest eigenpairs
of $B$. Geodesics are not exactly Euclidean, so a negative eigenvalue is
clipped to zero. `eigenvalues` holds $\Lambda_n$; they equal
`sklearn.manifold.Isomap().kernel_pca_.eigenvalues_`.

**Landmark Isomap** (de Silva & Tenenbaum, 2003; 2004). $m$ landmarks
$\ell_1, \dots, \ell_m$ are drawn uniformly without replacement (seeded by
`random_state`). Only $\Delta \in \mathbb R^{m \times N}$, the geodesics from
the landmarks, is computed. With $\Delta_m$ its landmark columns,

$$
B_m = -\tfrac12 J \Delta_m^{(2)} J = V \Lambda V^\top,\qquad
y_a = -\tfrac12\, \Lambda_n^{-1/2} V_n^\top \big(\delta_a - \bar\delta\big),
$$

where $\delta_a = \Delta^{(2)}_{\cdot a}$ holds point $a$'s squared geodesic
distances to the landmarks and $\bar\delta$ is the mean column of
$\Delta_m^{(2)}$. A landmark lands exactly on its MDS coordinate
$\Lambda_n^{1/2} V_n^\top e_j$ (because $V_n \perp \mathbf 1$), so with every
point a landmark this is Isomap. A zero eigenvalue gets a zero coordinate.

| | Isomap | Landmark Isomap |
|---|---|---|
| Shortest paths | $O(N^2 \log N + N^2 k)$, $N \times N$ dense | $O(m N \log N + m N k)$, $m \times N$ |
| Eigenproblem | $O(N^3)$ | $O(m^3)$ |
| Placement | — | $O(m n N)$ |

## Pseudocode

```text
fit(X):
    G      ← knn_graph(X, k, weighting="connectivity", ensure_connected=True)
    len_e  ← ‖x_s − x_r‖ for each edge (s, r)            # duplicates: tiny > 0
    if n_landmarks is None:
        D  ← dijkstra(G, len)                            # N × N, CPU
        B  ← −½ (D² − row means − column means + grand mean)
        (λ, U) ← top n eigenpairs of B;  λ ← max(λ, 0)
        Y  ← U · sqrt(λ)
    else:
        L  ← m random distinct indices (key = random_state)
        Δ  ← dijkstra(G, len, sources = L)               # m × N
        B_m ← −½ J (Δ²[:, L]) J;  (λ, V) ← top n eigenpairs;  λ ← max(λ, 0)
        Y  ← −½ (Δ² − mean column of Δ²[:, L])ᵀ · V / sqrt(λ)
    return Y, λ
```

## References

- Tenenbaum, J. B., de Silva, V. & Langford, J. C. (2000). A global geometric
  framework for nonlinear dimensionality reduction. *Science*, 290(5500),
  2319–2323. [doi:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319)
- de Silva, V. & Tenenbaum, J. B. (2003). Global versus local methods in
  nonlinear dimensionality reduction. *Advances in Neural Information
  Processing Systems 15*, 721–728.
- de Silva, V. & Tenenbaum, J. B. (2004). *Sparse multidimensional scaling
  using landmark points*. Technical report, Stanford University.
- Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
  *Psychometrika*, 17(4), 401–419.
  [doi:10.1007/BF02288916](https://doi.org/10.1007/BF02288916)

## API

::: manipy.Isomap
