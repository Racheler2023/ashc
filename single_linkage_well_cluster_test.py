import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import real_data
import linkage
import math
import os
from sklearn.cluster import DBSCAN
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False  # Already connected
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1
        return True

def calculate_cluster_metrics(data, clusters):

    intra_max_list = []
    inter_min_list = []

    print(len(clusters))
    non_empty_clusters = [c for c in clusters if len(c) > 0]
    

    for cluster in non_empty_clusters:
        m = len(cluster)
        edges = []

        for i in range(m):
            for j in range(i+1, m):
                dist = np.linalg.norm(data[cluster[i]] - data[cluster[j]])
                edges.append((dist, i, j))
        
        # Kruskal algorithm to find the maximum edge of MST
        edges.sort()
        uf = UnionFind(m)
        max_edge = 0.0
        edges_added = 0
        for dist, u, v in edges:
            if uf.union(u, v):
                max_edge = max(max_edge, dist)
                edges_added += 1
                if edges_added == m-1:
                    break
        intra_max_list.append(max_edge)
    
    # Calculate the minimum distance from each non-empty cluster to other clusters
    for i, current_cluster in enumerate(non_empty_clusters):
        global_min = float('inf')
        # Only traverse other non-empty clusters
        for j, other_cluster in enumerate(non_empty_clusters):
            if i == j:
                continue
            # Calculate the minimum distance between two clusters (optimized matrix operation)
            current_data = data[current_cluster]
            other_data = data[other_cluster]
            # Use broadcast to calculate the distance between all points and take the minimum value
            dist_matrix = np.sqrt(((current_data[:, np.newaxis] - other_data) ** 2).sum(axis=2))
            min_dist = dist_matrix.min()
            if min_dist < global_min:
                global_min = min_dist
        # print(global_min)
        inter_min_list.append(global_min if global_min != float('inf') else 0.0)
    
    # Process empty clusters in the original clusters (add 0.0 placeholders)
    original_cluster_count = len(clusters)
    non_empty_count = len(non_empty_clusters)
    if non_empty_count < original_cluster_count:
        # If there is an empty cluster, add the default value
        intra_max_list += [0.0] * (original_cluster_count - non_empty_count)
        inter_min_list += [0.0] * (original_cluster_count - non_empty_count)
    
    return intra_max_list, inter_min_list

# Output function remains unchanged
def save_results(k, intra_max_list, inter_min_list, filename="clusters_by_DBSCAN.txt"):
    with open(filename, 'a') as f:
        f.write(f"===Yeast cluster number k={k} ===\n")
        for cluster_id, (intra, inter) in enumerate(zip(intra_max_list, inter_min_list)):
            f.write(f"Clus1ter {cluster_id}: Intra-cluster MaxIntra={intra:.4f}, Inter-cluster MinInter={inter:.4f}\n")
        f.write("\n")


if __name__ == "__main__":
    
    data, _ = real_data.get_processed_yeast_data()  
    
    #DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(data)
    unique_labels = np.unique(labels)

    # Build clusters with index as content
    clusters = {
        label: np.where(labels == label)[0].tolist()
        for label in unique_labels
    }

    
    noise_points = clusters.get(-1, [])

    # Remove noise points and keep clusters of non-noise points
    clusters_without_noise = [
        cluster
        for label, cluster in clusters.items()
        if label != -1
    ]


    intra_max_list, inter_min_list = calculate_cluster_metrics(data, clusters_without_noise)

    save_results(len(clusters_without_noise), intra_max_list, inter_min_list)

