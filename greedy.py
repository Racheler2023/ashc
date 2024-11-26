def choose_next_center(tree, leaves, leaf_to_nearest_center, leaf_to_shortest_path_sum):
    min_path_sum = float('inf')
    next_center = None
    for leaf in leaves:
        # Consider only leaf nodes that have not been selected as centers
        if leaf.center_id is None:
            path_sum = 0
            for other_leaf in leaves:
                 path_weight=tree.find_shortest_path_weight(other_leaf, leaf)
                 if leaf_to_nearest_center[other_leaf] is None:
                     path_sum += path_weight
                 elif path_weight > leaf_to_shortest_path_sum[other_leaf]:
                        path_sum += leaf_to_shortest_path_sum[other_leaf]
                 else:
                     path_sum += path_weight
            if path_sum < min_path_sum:
                min_path_sum = path_sum
                next_center = leaf
    return next_center


def k_median_dec(tree, num_centers, deleted_index=None):
    leaves = tree.collect_leaf_nodes()
    if deleted_index:
        leaves = [item for item in leaves if item.point_ids[0] not in deleted_index]
    leaf_to_nearest_center = {leaf: None for leaf in leaves}
    leaf_to_shortest_path_sum = {leaf: float('inf') for leaf in leaves}

    all_clusters = []
    centers = []
    all_costs = []
    for _ in range(num_centers):
        center = choose_next_center(tree, leaves, leaf_to_nearest_center, leaf_to_shortest_path_sum)
        if center:
            center.center_id = center.point_ids
            centers.append(center)
            cost=0
            for leaf in leaves:
                path_weight = tree.find_shortest_path_weight(leaf, center)
                if leaf_to_nearest_center[leaf] is None or path_weight < leaf_to_shortest_path_sum[leaf]:
                    leaf_to_nearest_center[leaf] = center
                    leaf_to_shortest_path_sum[leaf] = path_weight
                cost+=leaf_to_shortest_path_sum[leaf]
            # Record clustering results
            clusters = []
            for center in centers:
                cluster = []
                cluster.extend(point_id for leaf, nearest_center in leaf_to_nearest_center.items() if nearest_center == center for point_id in leaf.point_ids)
                clusters.append(cluster)
            all_clusters.append(clusters)
            all_costs.append(cost)

    # Clean up the markers after completion
    tree.clear_node_marks()
    return all_clusters, all_costs
