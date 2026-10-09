# scikit-learn adapters

`manipy.sklearn` wraps manipy's estimators in scikit-learn's conventions
(mutable, `fit` returns `self`, fitted attributes end in `_`, NumPy in and
out). It needs the extra:

```bash
pip install "manipy-jax[sklearn]"
```

The core never imports scikit-learn; `import manipy` does not load it.

## Embeddings

The embedding adapters are transductive, like
`sklearn.manifold.SpectralEmbedding`: `fit` and `fit_transform`, no
`transform`. They pass scikit-learn's `check_estimator`, and on small inputs
cap `n_neighbors` at `n_samples - 1` and `n_components` at what the data
allow.

```python
from manipy.sklearn import Isomap

Y = Isomap(n_components=2, n_neighbors=12).fit_transform(X)
```

::: manipy.sklearn.Isomap

::: manipy.sklearn.LocallyLinearEmbedding

::: manipy.sklearn.DiffusionMaps

## Out-of-sample extension

`NystromExtension` turns a transductive embedding into a full transformer:
it fits the wrapped estimator and adds a Nyström `transform`, so the
embedding can sit inside a `Pipeline`. The training points come back
exactly. The maths is on the [Out-of-sample extension](out_of_sample.md)
page.

```python
import kernellib.sklearn as kls
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from manipy.sklearn import NystromExtension

clf = make_pipeline(
    NystromExtension(kls.LaplacianEigenmaps(n_components=10, n_neighbors=12)), SVC()
)
clf.fit(X_train, y_train).predict(X_test)
```

::: manipy.sklearn.NystromExtension

## Manifold alignment

Alignment is a **partial fit** of the scikit-learn contract. It needs several
domains with different numbers of features, which one
`(n_samples, n_features)` array cannot hold, so `fit` takes *lists* of
per-domain arrays and `transform` needs the domain:

```python
from sklearn.svm import SVC
from manipy.sklearn import ManifoldAlignment

ma = ManifoldAlignment(n_components=10).fit([X_hymap, X_aviris], [y_hymap, y_aviris])
svm = SVC().fit(ma.transform(X_hymap_labelled, domain=0), y_hymap_labelled)
pred = svm.predict(ma.transform(X_aviris, domain=1))  # the other sensor
```

What works: `clone`, `get_params` / `set_params`, `check_is_fitted`, NumPy
output. What does not: `fit_transform`, `Pipeline`, `GridSearchCV` (they pass
one array), and `check_estimator`. The adapter is tested directly instead.
The maths is on the [Manifold alignment](alignment.md) page.

::: manipy.sklearn.ManifoldAlignment
