# DBSCAN: density-based clustering

Intermediate | classical-ml | unsupervised | clustering | dbscan

### The problem, from first principles

K-Means (`01`/`02`) and GMM (`04`/`05`) both need to be told `k`, how many clusters to look for, and both assume every point belongs to some cluster. Real data doesn't always cooperate: sometimes the right number of clusters isn't known ahead of time, and sometimes a handful of points genuinely are outliers that shouldn't be forced into any cluster at all.

### From theory to code

Implement `region_query(input, point_idx, eps)`, the indices of every point within `eps` of a given point (itself included), and `dbscan_fit(input, eps, min_samples)`, which grows clusters outward from dense regions and marks the rest as noise. Reuse `01-knn`'s `pairwise_distances`. `-1` means "noise," never a real cluster id — real cluster ids start at `0`.

### Constraints

- `region_query` returns indices where distance is `<= eps` (not strictly `<`), always including the query point itself.
- `dbscan_fit` returns one cluster id per row of `input`, or `-1` for noise.
- A point needs `>= min_samples` neighbors (including itself) to be a "core" point that can start or extend a cluster.
- A "border" point (within `eps` of a core point, but not itself core) joins that cluster without extending it further.
- Cluster ids are assigned in the order clusters are discovered, starting at `0`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

A point with fewer than `min_samples` neighbors doesn't necessarily stay noise forever — it might still get absorbed later as a border point of some other, later-discovered cluster. Don't treat "not enough neighbors right now" as a final decision.

</details>

<details>
<summary>Hint 2</summary>

Grow a cluster with a `seeds` list processed like a queue (append new neighbors to the end, advance an index rather than removing from the front) — but only extend the search through neighbors that are _themselves_ core points; a border point joins the cluster without growing it further.

</details>
