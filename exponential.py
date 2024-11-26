import numpy as np
import math
import random
import concurrent.futures
from functools import partial


# Randomize by seed
def sample_from_p(p, seed):
    rng = np.random.default_rng(seed=seed)
    p = np.array(p)
    while True:
        j = rng.choice(p.size, 6666)
        t = rng.random(j.size)
        k = np.where(t < p[j])[0]
        if k.size == 0:
            continue
        else:
            j = j[k[0]]
            break
    return j

# Index mechanism algorithm implementation:
def choose_next_center_ran(tree, leaves, leaf_to_nearest_center, leaf_to_shortest_path_sum, epsilon, seed,
                           deleted_index, center_set):
    total_cost = 0
    cost_distribution = []
    _min = float('inf')
    for leaf in leaves:
        if leaf.point_ids[0] not in center_set:
            path_sum = 0
            for other_leaf in leaves:
                path_weight = tree.find_shortest_path_weight(other_leaf, leaf)
                if leaf_to_nearest_center[other_leaf] is None:
                    path_sum += path_weight
                elif path_weight > leaf_to_shortest_path_sum[other_leaf]:
                    path_sum += leaf_to_shortest_path_sum[other_leaf]
                else:
                    path_sum += path_weight

            cost_distribution.append((leaf, path_sum))
            if _min > path_sum:
                _min = path_sum
            total_cost += path_sum

    if total_cost == 0:
        remaining_leaves = [leaf for leaf in leaves if leaf.point_ids[0] not in center_set]
        return random.choice(remaining_leaves) if remaining_leaves else None
    # lambda parameter setting
    n = len(leaves)

    lambda_value = 3 * math.log2(n) / (_min * epsilon)

    probabilities = [(leaf, math.exp(-lambda_value * cost)) for leaf, cost in cost_distribution]

    total_probability = sum(prob for _, prob in probabilities)
    probabilities = [(leaf.point_ids[0], prob / total_probability) for leaf, prob in probabilities]
    if deleted_index:
        for index in deleted_index:
            probabilities.append((index, 0))
    probabilities = sorted(probabilities, key=lambda x: x[0])

    p = [prob for _, prob in probabilities]
    index = sample_from_p(p, seed)
    chosen_center = probabilities[index][0]

    for leaf in leaves:
        if leaf.point_ids[0] == chosen_center:
            return leaf


def k_median_ranexp(tree, num_centers, epsilon, seed, deleted_index=None):
    center_set = set()
    leaves = tree.collect_leaf_nodes()
    if deleted_index:
        leaves = [item for item in leaves if item.point_ids[0] not in deleted_index]
    leaf_to_nearest_center = {leaf: None for leaf in leaves}
    leaf_to_shortest_path_sum = {leaf: float('inf') for leaf in leaves}

    centers = []
    all_costs = []
    all_clusters = []
    for _ in range(num_centers):
        center = choose_next_center_ran(tree, leaves, leaf_to_nearest_center, leaf_to_shortest_path_sum, epsilon, seed,
                                        deleted_index, center_set)
        if center:
            center_set.add(center.point_ids[0])

            centers.append(center)

            cost = 0
            for leaf in leaves:
                path_weight = tree.find_shortest_path_weight(leaf, center)
                if leaf_to_nearest_center[leaf] is None or path_weight < leaf_to_shortest_path_sum[leaf]:
                    leaf_to_nearest_center[leaf] = center
                    leaf_to_shortest_path_sum[leaf] = path_weight
                cost += leaf_to_shortest_path_sum[leaf]
            # Record clustering results
            clusters = []
            for center in centers:
                cluster = []
                cluster.extend(
                    point_id for leaf, nearest_center in leaf_to_nearest_center.items() if nearest_center == center for
                    point_id in leaf.point_ids)
                clusters.append(cluster)
            all_clusters.append(clusters)
            all_costs.append(cost)

    return all_clusters, all_costs
