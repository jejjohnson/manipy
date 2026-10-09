# Metrics

`manipy.metrics` holds the classification metrics reported for HSI benchmarks
and the neighbourhood-preservation metrics for embeddings.

## Formulation

With the confusion matrix $C$ ($C_{ij}$ pixels of true class $i$ predicted as
$j$), $n = \sum_{ij} C_{ij}$, $C_{i+}$ and $C_{+j}$ the row and column sums:

- overall accuracy $\mathrm{OA} = \operatorname{tr}C / n$;
- per-class accuracy $C_{ii}/C_{i+}$ and average accuracy $\mathrm{AA}$, their
  mean over the classes present;
- Cohen's kappa $\kappa = (p_o - p_e)/(1 - p_e)$ with $p_o = \operatorname{tr}C/n$
  and $p_e = \sum_i C_{i+}C_{+i}/n^2$, and its large-sample variance (Congalton
  and Green, 2009; formula in `cohen_kappa_variance`).

With $R_X(i, j)$ the rank of $j$ among the neighbours of $i$ in the original
space $X$ (nearest is 1), $R_Y$ likewise in the embedding $Y$, and
$N_k^X(i)$ the $k$ nearest neighbours of $i$:

$$T(k) = 1 - \frac{2}{nk(2n-3k-1)}\sum_i\sum_{\substack{j \in N_k^Y(i)\\ j \notin N_k^X(i)}}
\big(R_X(i,j) - k\big),$$

continuity $C(k)$ is $T(k)$ with $X$ and $Y$ swapped, and

$$\mathrm{kNN}(k) = \frac{1}{nk}\sum_i |N_k^X(i)\cap N_k^Y(i)|,\qquad
\mathrm{LCMC}(k) = \mathrm{kNN}(k) - \frac{k}{n-1}.$$

## Pseudocode

```text
quality(X, Y, k):
    R_X, R_Y <- argsort(argsort(squared distances)), self at rank 0   # O(n^2)
    trustworthiness: sum (R_X - k) over pairs with R_Y in 1..k and R_X > k
    continuity:      sum (R_Y - k) over pairs with R_X in 1..k and R_Y > k
    kNN, LCMC:       count pairs with R_X and R_Y both in 1..k
```

The rank matrices are dense, so these suit evaluation sets of a few thousand
points; subsample larger images.

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales.
  *Educational and Psychological Measurement* 20(1), 37-46.
- Congalton, R. G. and Green, K. (2009). *Assessing the Accuracy of Remotely
  Sensed Data: Principles and Practices*, 2nd ed. CRC Press.
- Venna, J. and Kaski, S. (2001). Neighborhood preservation in nonlinear
  projection methods: an experimental study. *ICANN 2001*, 485-491.
- Chen, L. and Buja, A. (2009). Local multidimensional scaling for nonlinear
  dimension reduction, graph drawing, and proximity analysis. *JASA* 104(485),
  209-219.

## API

::: manipy.metrics
    options:
      members: true
