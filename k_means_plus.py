import numpy as np


def kmeans_plus_plus_with_multiple_clusters(X, k, delete_index=None):
    np.random.seed(555)
    # If there are deleted points, filter X and keep the valid indexes
    if delete_index is not None:
        delete_index = np.array(list(delete_index)).astype(int)
        X_filtered = np.delete(X, delete_index, axis=0)
        original_indices = np.arange(len(X))
        valid_indices = np.delete(original_indices, delete_index)
    else:
        X_filtered = X
        valid_indices = np.arange(len(X))

    n_samples, n_features = X_filtered.shape
    centers = np.zeros((k, n_features))
    all_clusters = []
    all_centers = []

    # Randomly select the first center
    centers[0] = X_filtered[np.random.randint(n_samples)]
    all_centers.append(centers[0].tolist())
    all_clusters.append([list(valid_indices)])

    for i in range(1, k):
        distances = np.min(np.linalg.norm(X_filtered[:, np.newaxis] - centers[:i], axis=2), axis=1)
        prob = distances / distances.sum()
        cumulative_prob = np.cumsum(prob)
        rand = np.random.rand()
        center_index = np.searchsorted(cumulative_prob, rand)

        centers[i] = X_filtered[center_index]
        all_centers.append(centers[i].tolist())

        cluster_assignments = np.argmin(np.linalg.norm(X_filtered[:, np.newaxis] - centers[:i + 1], axis=2), axis=1)

        new_clusters = [[] for _ in range(i + 1)]
        for j, label in enumerate(cluster_assignments):
            actual_index = valid_indices[j]
            new_clusters[label].append(actual_index)

        all_clusters.append(new_clusters)

    return all_clusters, all_centers



