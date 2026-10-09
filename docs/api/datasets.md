# Datasets

`manipy.datasets` has keyed synthetic manifolds (JAX, no extra needed) and
loaders for three hyperspectral benchmark scenes.

The scene loaders need the `data` extra, `pip install "manipy-jax[data]"`
(`pooch` downloads and caches the files, `scipy.io` reads the MATLAB
containers). Both are imported inside the loaders, never by
`import manipy`. Scenes come from the Grupo de Inteligencia Computacional
(EHU) mirror.

## Formulation

The sheets are 2-D parameter domains $(t, v)$ mapped into 3-D:

- Swiss roll: $(t\cos t,\ 21v,\ t\sin t)$, $t \in [1.5\pi, 4.5\pi]$;
- S-curve: $(\sin t,\ 2v,\ \operatorname{sign}(t)(\cos t - 1))$,
  $t \in [-1.5\pi, 1.5\pi]$;
- severed sphere: the unit sphere restricted to polar angles
  $[\pi/8, 7\pi/8]$ and azimuths $[0, 2\pi - 0.55]$, uniform in area.

## Pseudocode

```text
loader(name, data_dir):
    for (image, ground-truth) file:
        path <- pooch.retrieve(url, known_hash, cache=data_dir)  # cached after the first call
        array <- scipy.io.loadmat(path)[variable]
    return image as float32 (H, W, bands), labels as int32 (H, W)  # 0 = unlabelled
```

## References

- Roweis, S. T. and Saul, L. K. (2000). Nonlinear dimensionality reduction
  by locally linear embedding. *Science* 290(5500), 2323-2326.
- Grupo de Inteligencia Computacional (UPV/EHU), Hyperspectral Remote
  Sensing Scenes. <https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes>

## API

::: manipy.datasets
    options:
      members: true
