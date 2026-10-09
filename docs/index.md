# manipy

> Manifold alignment, dimensionality reduction and hyperspectral workflows in
> JAX, built on kernellib.

`manipy` owns the methods that sit above kernellib's graph and kernel
machinery: manifold alignment across domains, non-spectral-graph embeddings
(Isomap, the LLE family, diffusion maps), out-of-sample extension,
hyperspectral (HSI) workflows, embedding-quality and classification metrics,
dataset loaders, and scikit-learn adapters for all of them.

It does **not** own graphs, Laplacians, Laplacian / Schrödinger eigenmaps, LPP
or kernel PCA. Those live in
[kernellib](https://github.com/jejjohnson/kernellib), and manipy uses them
without re-exporting them: import `kernellib.LaplacianEigenmaps` directly.

:::{warning} Status: pre-alpha (phase M1)
This repository was reset from a 2018 numpy / scikit-learn package to a fresh
JAX package. Manifold alignment
([`ManifoldAlignment`](xref:api#manipy.ManifoldAlignment): Wang, SSMA, SEMA)
and its scikit-learn adapter are in; the other algorithms land in the phases
of the [roadmap](roadmap/roadmap.md).
:::

## Installation

The PyPI distribution is named `manipy-jax` (the name `manipy` belongs to an
unrelated project); the import name is `manipy`. It is not published yet, and
it depends on kernellib and gaussx, which are pinned by git tag. Until
release, install from the repository:

::::{tab-set}
:::{tab-item} uv

```bash
uv add "manipy-jax @ git+https://github.com/jejjohnson/manipy.git"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/manipy.git
cd manipy
make install
```
:::
::::

The scikit-learn adapters in `manipy.sklearn` need the optional extra,
`manipy-jax[sklearn]`.

The 2018 code is preserved on the `legacy` branch and the `v0.0.0-legacy` tag.

## Where to go next

| I want to… | Start here |
|---|---|
| See what is planned, and in which order | [Roadmap](roadmap/roadmap.md) |
| Read the API | [API reference](xref:api#manipy) |
| Contribute | [Contributing](contributing.md) |
