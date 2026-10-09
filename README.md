# manipy

Manifold alignment, dimensionality reduction and hyperspectral workflows in
JAX, built on [kernellib](https://github.com/jejjohnson/kernellib) and
[gaussx](https://github.com/jejjohnson/gaussx).

> **Status: pre-alpha (roadmap phase M1).** Manifold alignment
> (`ManifoldAlignment`: Wang, SSMA, SEMA) and its scikit-learn adapter are
> in. The other algorithms land in phases M2 to M5; see the
> [roadmap](https://github.com/jejjohnson/kernellib/blob/main/docs/roadmap/roadmap-manipy.md).

## Legacy code

manipy was first a 2018 numpy / scikit-learn package (`manilearn/`). That code
is preserved on the [`legacy`](https://github.com/jejjohnson/manipy/tree/legacy)
branch and the `v0.0.0-legacy` tag. Nothing from it is carried forward: its
algorithms (Laplacian eigenmaps, LPP, Schrödinger eigenmaps) now live in
kernellib, and the rest is rebuilt here from the reference implementations.

## What lives where

| In manipy | In kernellib |
|---|---|
| manifold alignment, Isomap, the LLE family, diffusion maps, out-of-sample extension | graphs, Laplacians, graph kernels |
| hyperspectral (HSI) workflows, embedding-quality and classification metrics, dataset loaders | Laplacian / Schrödinger eigenmaps, LPP, SEP, kernel PCA |
| scikit-learn adapters (`manipy.sklearn`) for the above | kernels, dependence measures, kernel regression |

manipy uses kernellib and never re-exports it: import
`kernellib.LaplacianEigenmaps` directly.

## Installation

The PyPI distribution is **`manipy-jax`** (the name `manipy` belongs to an
unrelated project); the import name is `manipy`. It is not published yet, and
kernellib and gaussx are pinned by git tag, so install from the repository:

```bash
uv add "manipy-jax @ git+https://github.com/jejjohnson/manipy.git"
```

For development:

```bash
git clone https://github.com/jejjohnson/manipy.git
cd manipy
make install     # uv sync --all-groups + pre-commit hooks
make test        # fast tier
make docs        # needs mystmd: npm install -g mystmd
```

The scikit-learn adapters need the optional extra, `manipy-jax[sklearn]`; the
core never imports scikit-learn.

```python
import manipy

manipy.__version__  # "0.0.0"
```

## Development

See [CLAUDE.md](CLAUDE.md) for the package layout, conventions and test tiers,
[AGENTS.md](AGENTS.md) for standing agent instructions, and
[CONTRIBUTING.md](CONTRIBUTING.md). Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/).

## License

MIT; see [LICENSE](LICENSE).
