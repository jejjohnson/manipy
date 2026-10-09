# Manifold alignment

Several domains (for example two hyperspectral sensors with different
bands) see the same kind of scene in different feature spaces
$\mathbb R^{D_1}, \dots, \mathbb R^{D_m}$. Manifold alignment finds one linear
projection per domain into a shared $n$-dimensional space in which each
domain keeps its own neighbourhood geometry, samples of the same class meet
across domains, and (SSMA) different classes are pushed apart. A classifier
trained on one domain's embedding then applies to every other domain.

```python
import manipy

ma = manipy.ManifoldAlignment(method="ssma", n_components=15, mu=0.5)
ma = ma.fit([X_hymap, X_aviris], [y_hymap, y_aviris])  # -1 marks unlabelled pixels
Z_train = ma.transform(X_hymap_labelled, domain=0)
Z_test = ma.transform(X_aviris_scene, domain=1)  # the other sensor, same space
```

The graphs come from [kernellib](https://jejjohnson.github.io/kernellib/)
(`knn_graph`, `spatial_spectral_graph`, `grid_graph`) and the eigenproblem
is solved by `gaussx.eigh_generalized`. manipy does not re-export either.

## Formulation

Domain $i$ has data $X_i \in \mathbb R^{N_i \times D_i}$ and labels
$y_i \in \{-1, 0, 1, \dots\}^{N_i}$ ($-1$: unlabelled); class $c$ means the
same thing in every domain. With $\bar X_i$ the domain centred at its mean
$\mu_i$:

- $Z = \operatorname{blkdiag}(\bar X_1, \dots, \bar X_m)$, of size
  $\Sigma N \times \Sigma D$;
- $W_g = \operatorname{blkdiag}(W_1, \dots, W_m)$, $W_i$ the k-NN graph of
  $X_i$ (no cross-domain edges), with Laplacian $L_g$ and degrees $D_g$;
- $W_s[a, b] = 1$ if $y_a = y_b$ and $W_d[a, b] = 1$ if $y_a \ne y_b$, over
  all labelled samples of all domains, with Laplacians $L_s$ and $L_d$;
- for SEMA, $P = \operatorname{blkdiag}(P_1, \dots, P_m)$, $P_i$ the
  Laplacian of `spatial_spectral_graph(X_i, S_i)` on the domain's spatial
  graph $S_i$.

The stacked projection $F = [F_1; \dots; F_m] \in \mathbb R^{\Sigma D \times n}$
holds the $n$ smallest solutions of

$$
(A + \lambda_r Z^\top Z)\, f = \lambda\, (B + \lambda_r Z^\top Z)\, f,
$$

| `method` | $A$ | $B$ | Reference |
|---|---|---|---|
| `"wang"` | $Z^\top(L_g + \mu L_s)Z$ | $Z^\top D_g Z$ | Wang & Mahadevan (2011) |
| `"ssma"` | $Z^\top((1-\mu) L_g + \mu\, s_s L_s)Z$ | $s_d\, Z^\top L_d Z$ | Tuia et al. (2014) |
| `"sema"` | $Z^\top((1-\mu)(L_g + \tilde\alpha P) + \mu L_s)Z$ | $Z^\top D_g Z$ | Schrödinger alignment, with the potential of Cahill et al. (2014) |

- **SSMA rescaling.** Tuia's code adds the identity to each class graph and
  rescales it to the total weight of $W_g$. The self-loops cancel in
  $L = D - W$, so this is the scalar
  $s_s = \mathbf 1^\top W_g \mathbf 1 / (\sum_c n_c^2 + \Sigma N)$ on $L_s$
  and $s_d = \mathbf 1^\top W_g \mathbf 1 / (\ell^2 - \sum_c n_c^2 + \Sigma N)$
  on $L_d$, with $n_c$ the labelled samples of class $c$ and
  $\ell = \sum_c n_c$.
- **Potential.** $\tilde\alpha = \alpha \operatorname{tr}(L_g) /
  \operatorname{tr}(P)$ with `normalize_potential=True` (the default), else
  $\alpha$ (the MATLAB's behaviour).
- **Ridge.** $\lambda_r = r \operatorname{tr}(B) / \operatorname{tr}(Z^\top Z)$
  for `ridge` $= r$: $Z^\top Z$ divided by its mean eigenvalue
  $\operatorname{tr}(Z^\top Z)/\Sigma D$, times $r$ times $B$'s mean
  eigenvalue $\operatorname{tr}(B)/\Sigma D$. It matches the MATLAB's
  $\lambda I$ added in sample space before projection, and is scale free.
  `ridge=0` solves the pencil exactly through gaussx's singular-$B$ path.
- **Out of sample.** $\operatorname{transform}(x, i) = (x - \mu_i) F_i$, with
  $F_i$ rows $o_i{:}o_{i+1}$ of $F$ at the cumulative offsets
  $o_i = \sum_{j < i} D_j$. With `standardize=True` each domain's embedding
  is then z-scored with the mean and standard deviation of its labelled
  embeddings, as Tuia's code did before classification.

`"sema"` with $\alpha = 0$ is `"wang"` with $\mu' = \mu / (1 - \mu)$, its
eigenvalues scaled by $1 - \mu$: $A_{\text{sema}} = (1-\mu) A_{\text{wang}}$
and $B$ is the same.

## Efficient assembly

Nothing of size $\Sigma N \times \Sigma N$ is formed.

- The geometric terms are block diagonal: $Z^\top L_g Z =
  \operatorname{blkdiag}_i(\bar X_i^\top L_i \bar X_i)$ through each graph's
  sparse `laplacian_operator()`, likewise $Z^\top D_g Z$ and $Z^\top P Z$.
- The class terms are in closed form over the $\ell$ labelled rows $Z_\ell$.
  For the complete graph $K_c$ on class $c$,
  $Z^\top L_{K_c} Z = n_c Z_c^\top Z_c - (Z_c^\top \mathbf 1)(Z_c^\top \mathbf 1)^\top$,
  so $Z^\top L_s Z = Z_\ell^\top \operatorname{diag}(n_{y}) Z_\ell - S S^\top$
  with $S$ the per-class column sums, and
  $Z^\top L_d Z = Z^\top L_{K_\ell} Z - Z^\top L_s Z$.
- Cost: $O(\ell (\Sigma D)^2)$ time and $O((\Sigma D)^2)$ memory for the
  class terms (the MATLAB used $O(\ell^2)$), $O(\sum_i |E_i| D_i + N_i D_i^2)$
  for the geometric ones, and $O((\Sigma D)^3)$ for the eigenproblem, whatever
  the number of pixels.

## Pseudocode

```text
fit(X_1..X_m, y_1..y_m, [S_1..S_m]):
    for each domain i:
        μ_i  ← mean of X_i;  X̄_i ← X_i − μ_i
        W_i  ← knn_graph(X_i, k)
        G_i  ← X̄_iᵀ L_i X̄_i;   H_i ← X̄_iᵀ D_i X̄_i;   C_i ← X̄_iᵀ X̄_i
        (sema) V_i ← X̄_iᵀ P_i X̄_i,  P_i ← Laplacian(spatial_spectral_graph(X_i, S_i))
    ZᵀL_gZ, ZᵀD_gZ, ZᵀZ, ZᵀPZ ← block-diagonal stacks of G, H, C, V
    Z_ℓ ← labelled rows of every X̄_i, placed at columns o_i : o_i + D_i
    ZᵀL_sZ ← Z_ℓᵀ diag(n_y) Z_ℓ − S Sᵀ;   ZᵀL_dZ ← ℓ Z_ℓᵀZ_ℓ − (Z_ℓᵀ1)(Z_ℓᵀ1)ᵀ − ZᵀL_sZ
    A, B ← table above;   λ_r ← r · tr(B) / tr(ZᵀZ)
    (λ, F) ← eigh_generalized(A + λ_r ZᵀZ, B + λ_r ZᵀZ, rank=n), smallest first
    F_i ← F[o_i : o_i + D_i]            # cumulative offsets o_i = D_1 + … + D_{i−1}

transform(x, i):  (x − μ_i) F_i,  then z-score with the domain's labelled statistics if standardize
```

## Fixes to the original MATLAB

The [legacy MATLAB](https://github.com/jejjohnson/manifold_learning) is a
numerical reference only. Differences, all deliberate:

- **Cumulative offsets.** `manifoldalignmentprojections.m` sliced the
  eigenvectors with non-cumulative indices (`D_i + 1 : D_i + D_{i+1}`), which
  is right for two domains only.
- **Wang's $A$.** The MATLAB's `'wang'` branch used SSMA's
  $(1-\mu)L + \mu L_s$; this follows Wang & Mahadevan, $L_g + \mu L_s$.
- **Ridge on both sides.** The MATLAB added $\lambda I$ to $A$ only for
  `'wang'` and `'sema'`; here the same relative ridge goes on both sides for
  every method.
- **Centring.** Domains are centred before projection, and `transform`
  removes the same means.
- **Class graphs in closed form**, $O(\ell (\Sigma D)^2)$ instead of dense
  $\Sigma N \times \Sigma N$ label comparisons.
- **Labels.** `-1` marks unlabelled samples, so class `0` is a real class (the
  MATLAB used `0` for unlabelled).
- **Potential normalisation.** SEMA's potential is trace-normalised by
  default; `normalize_potential=False` restores the MATLAB's raw $\alpha$.
- **Standardisation.** The MATLAB z-scored test embeddings with
  `repmat(…, 2*T, 1)`, `T = length(XTest)/2`, and `length` is the larger
  dimension, so it broke when a test set had fewer samples than features;
  here the statistics broadcast.

## References

- Wang, C. & Mahadevan, S. (2011). Heterogeneous domain adaptation using
  manifold alignment. *Proceedings of IJCAI 2011*, 1541–1546.
  [doi:10.5591/978-1-57735-516-8/IJCAI11-259](https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-259)
- Tuia, D., Volpi, M., Trolliet, M. & Camps-Valls, G. (2014). Semisupervised
  manifold alignment of multimodal remote sensing images. *IEEE Transactions
  on Geoscience and Remote Sensing*, 52(12), 7708–7720.
  [doi:10.1109/TGRS.2014.2317499](https://doi.org/10.1109/TGRS.2014.2317499)
- Cahill, N. D., Czaja, W. & Messinger, D. W. (2014). Schroedinger eigenmaps
  with nondiagonal potentials for spatial-spectral clustering of
  hyperspectral imagery. *Proc. SPIE 9088*, 908804.
  [doi:10.1117/12.2050651](https://doi.org/10.1117/12.2050651)

## API

::: manipy.ManifoldAlignment
