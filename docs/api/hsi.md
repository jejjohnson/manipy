# Hyperspectral helpers

`manipy.hsi` is the plumbing around kernellib's embeddings for hyperspectral
images (HSI): cube to pixel-array conversion, pixel grids, image-level
Schrödinger potentials and stratified train / test splits. The embeddings
themselves (`kernellib.LaplacianEigenmaps`, `kernellib.SchrodingerEigenmaps`)
live in kernellib; import them from there.

## Formulation

An $H \times W \times D$ cube becomes $X \in \mathbb{R}^{N \times D}$,
$N = HW$, with pixel $(r, c)$ at row $rW + c$ (**row-major**, the node order
of `kernellib.GridGraph`). The 2016-2018 code used column-major order, so old
pixel indices do not carry over.

The spatial-spectral potential (Cahill, Czaja and Messinger, 2014) is the
Laplacian $V = D_w - W$ of the pixel grid with edge weights

$$w_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
\quad\text{for adjacent pixels } i \sim j,$$

and the partial-label potential is the Laplacian of the graph that joins every
pair of labelled pixels $i, j$ with $y_i = y_j$.

## Pseudocode

```text
stratified_split(y, fraction | count, key, background=0):
    for each class c != background:
        idx_c <- positions of y == c
        shuffle idx_c with fold_in(key, c)
        n_c   <- max(1, round(fraction * |idx_c|))  or  min(count, |idx_c|)
        train <- train + idx_c[:n_c];  test <- test + idx_c[n_c:]
    return sorted(train), sorted(test)
```

## References

- Cahill, N. D., Czaja, W. and Messinger, D. W. (2014). Schroedinger
  eigenmaps with nondiagonal potentials for spatial-spectral clustering of
  hyperspectral imagery. *Proc. SPIE* 9088.

## API

::: manipy.hsi
    options:
      members: true
