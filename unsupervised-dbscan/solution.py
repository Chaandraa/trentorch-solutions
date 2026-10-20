import numpy as np


def pairwise_distances(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """Computes pairwise Euclidean distances between rows in X and Y."""
    if Y is None:
        Y = X
    dot_product = np.dot(X, Y.T)
    x_square = np.sum(X**2, axis=1, keepdims=True)
    y_square = np.sum(Y**2, axis=1, keepdims=True)
    distances_sq = x_square + y_square.T - 2 * dot_product
    return np.sqrt(np.maximum(distances_sq, 0.0))


def region_query(input: np.ndarray, point_idx: int, eps: float) -> np.ndarray:
    """
    Returns:
        the indices of every row of input within eps (inclusive) of
        input[point_idx], including point_idx itself.
    """
    query_point = input[point_idx : point_idx + 1]
    distances = pairwise_distances(input, query_point).ravel()
    return np.where(distances <= eps)[0]


def dbscan_fit(input: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Fast vectorised DBSCAN that passes strict execution time limits.
    """
    n_samples = len(input)

    # 1. Precompute pairwise distance matrix once (O(N^2) instead of O(N^2 * iterations))
    dist_matrix = pairwise_distances(input)

    # 2. Precompute all point neighborhoods instantly using boolean masking
    # neighbors_mask[i] is a boolean array of length n_samples
    neighbors_mask = dist_matrix <= eps
    neighbor_counts = np.sum(neighbors_mask, axis=1)

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)

    cluster_id = 0

    for i in range(n_samples):
        if visited[i]:
            continue

        visited[i] = True

        # Check if point i is a core point
        if neighbor_counts[i] < min_samples:
            continue

        # Point i is core -> start new cluster
        labels[i] = cluster_id

        # Seed queue initialized with indices of point i's neighbors
        seeds = np.where(neighbors_mask[i])[0].tolist()
        
        # Track what has been added to seeds to prevent exponential queue growth
        in_seeds = np.zeros(n_samples, dtype=bool)
        in_seeds[seeds] = True

        seed_idx = 0
        while seed_idx < len(seeds):
            q = seeds[seed_idx]
            seed_idx += 1

            # Relabel noise points to border points
            if labels[q] == -1:
                labels[q] = cluster_id

            if visited[q]:
                continue

            visited[q] = True
            labels[q] = cluster_id

            # If q is also a core point, expand seeds queue without duplicates
            if neighbor_counts[q] >= min_samples:
                q_neighbors = np.where(neighbors_mask[q])[0]
                for neighbor in q_neighbors:
                    if not in_seeds[neighbor]:
                        in_seeds[neighbor] = True
                        seeds.append(neighbor)

        cluster_id += 1

    return labels
