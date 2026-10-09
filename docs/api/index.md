# API Reference

Everything listed in `manipy.__all__` is importable from the top level:

```python
import manipy

manipy.__version__
```

| Page | Contents |
|---|---|
| [Manifold alignment](alignment.md) | `ManifoldAlignment`: Wang, SSMA and SEMA alignment of several domains |
| [Isomap](isomap.md) | `Isomap`: Isomap and landmark Isomap |
| [Locally linear embedding](lle.md) | `LocallyLinearEmbedding`: LLE, modified LLE, Hessian LLE, LTSA |
| [Hyperspectral helpers](hsi.md) | `manipy.hsi`: pixel arrays, grids, potentials, stratified splits |
| [Metrics](metrics.md) | `manipy.metrics`: classification and embedding-quality metrics |
| [Datasets](datasets.md) | `manipy.datasets`: synthetic manifolds, Indian Pines, Pavia University, Salinas |
| [scikit-learn adapters](sklearn.md) | `manipy.sklearn` (extra `manipy-jax[sklearn]`) |

Modules appear here as each phase of the
[roadmap](https://jejjohnson.github.io/kernellib/roadmap-manipy/) lands.

## Package overview

::: manipy
    options:
      members: false
      show_root_heading: false
      show_source: false
