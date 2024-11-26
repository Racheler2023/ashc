import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.distance import cdist

def single_linkage(X, max_clusters, deleted_index=None):
    if deleted_index is not None:
        deleted_index = np.array(list(deleted_index)).astype(int)

        X_filtered = np.delete(X, deleted_index, axis=0)

        original_indices = np.arange(len(X))
        valid_indices = np.delete(original_indices, deleted_index)
    else:
        X_filtered = X
        valid_indices = np.arange(len(X))

    Z = hcluster.linkage(X_filtered, method='single')

    all_clusters = []
    all_costs = []
    for n_clusters in range(1, max_clusters + 1):

        labels = hcluster.fcluster(Z, t=n_clusters, criterion='maxclust')

        clusters = [[] for _ in range(n_clusters)]
        for point_index, label in enumerate(labels):
            if deleted_index is not None:
                actual_index = valid_indices[point_index]
            else:
                actual_index = point_index
            clusters[label - 1].append(actual_index)

        cost = compute_cost(X_filtered, labels)
        all_clusters.append(clusters)
        all_costs.append(cost)

    return all_clusters, all_costs
#The calculation cost here is not used
def compute_cost(X, labels):
    cost = 0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            distances = cdist(cluster_points, [center])
            cost += np.sum(distances)
    return cost

def complete_linkage(X, max_clusters, deleted_index=None):

    if deleted_index is not None:
        deleted_index = np.array(list(deleted_index)).astype(int)
        X_filtered = np.delete(X, deleted_index, axis=0)
        original_indices = np.arange(len(X))
        valid_indices = np.delete(original_indices, deleted_index)
    else:
        X_filtered = X
        valid_indices = np.arange(len(X))

    Z = hcluster.linkage(X_filtered, method='complete')

    all_clusters = []
    all_costs = []
    for n_clusters in range(1, max_clusters + 1):

        labels = hcluster.fcluster(Z, t=n_clusters, criterion='maxclust')

        clusters = [[] for _ in range(n_clusters)]
        for point_index, label in enumerate(labels):
            if deleted_index is not None:
                actual_index = valid_indices[point_index]
            else:
                actual_index = point_index
            clusters[label - 1].append(actual_index)

        cost = compute_cost(X_filtered, labels)
        all_clusters.append(clusters)
        all_costs.append(cost)

    return all_clusters, all_costs

def average_linkage(X, max_clusters, deleted_index=None):

    if deleted_index is not None:
        deleted_index = np.array(list(deleted_index)).astype(int)
        X_filtered = np.delete(X, deleted_index, axis=0)
        original_indices = np.arange(len(X))
        valid_indices = np.delete(original_indices, deleted_index)
    else:
        X_filtered = X
        valid_indices = np.arange(len(X))

    Z = hcluster.linkage(X_filtered, method='average')

    all_clusters = []
    all_costs = []
    for n_clusters in range(1, max_clusters + 1):

        labels = hcluster.fcluster(Z, t=n_clusters, criterion='maxclust')

        clusters = [[] for _ in range(n_clusters)]
        for point_index, label in enumerate(labels):
            if deleted_index is not None:
                actual_index = valid_indices[point_index]
            else:
                actual_index = point_index
            clusters[label - 1].append(actual_index)

        cost = compute_cost(X_filtered, labels)
        all_clusters.append(clusters)
        all_costs.append(cost)

    return all_clusters, all_costs

def ward_linkage(X, max_clusters, deleted_index=None):

    if deleted_index is not None:
        deleted_index = np.array(list(deleted_index)).astype(int)
        X_filtered = np.delete(X, deleted_index, axis=0)
        original_indices = np.arange(len(X))
        valid_indices = np.delete(original_indices, deleted_index)
    else:
        X_filtered = X
        valid_indices = np.arange(len(X))

    Z = hcluster.linkage(X_filtered, method='ward')

    all_clusters = []
    all_costs = []
    for n_clusters in range(1, max_clusters + 1):

        labels = hcluster.fcluster(Z, t=n_clusters, criterion='maxclust')

        clusters = [[] for _ in range(n_clusters)]
        for point_index, label in enumerate(labels):
            if deleted_index is not None:
                actual_index = valid_indices[point_index]
            else:
                actual_index = point_index
            clusters[label - 1].append(actual_index)

        cost = compute_cost(X_filtered, labels)
        all_clusters.append(clusters)
        all_costs.append(cost)

    return all_clusters, all_costs
