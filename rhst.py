import numpy as np
import math
from functools import lru_cache


class Rhst:
    class Node:
        def __init__(self, bounds, level, point_ids, parent=None, node_id=None):
            self.id = node_id
            self.bounds = bounds
            self.level = level
            self.point_ids = point_ids
            self.parent = parent
            self.children = []
            self.edge_weight = 0
            self.center_id = None

        def print_tree(self, node_id=None, indent=""):
            if node_id is None:
                node_id = self.id

            print(
                f"{indent}Node {self.id}: Level: {self.level}, Bounds: {self.bounds}, Edge Weight: {self.edge_weight}, Points: {self.point_ids}")

            for i, child in enumerate(self.children):
                child_id = self.id * 10 + (i + 1)
                child.print_tree(node_id=child_id, indent=indent + "  ")

        def clear_node_marks(self):
            if self is not None:
                # Reset the center node mark
                self.center_id = None
                for child in self.children:
                    # Modify the point space to satisfy the tree partitioning property
                    child.clear_node_marks()

    def __init__(self, original_data):
        root_bounds = self.point_process_random(original_data)
        self.root, current_id, max_level, max_id = self.build_2rhst_dec(root_bounds, original_data)
        self.split_to_max_level(self.collect_leaf_nodes(), max_level, original_data, max_id)
        self.dist = {}

    def print_tree(self):
        self.root.print_tree()

    def clear_node_marks(self):
        self.root.clear_node_marks()

    def point_process(self, original_data):
        original_data = np.array(original_data)
        n, d = original_data.shape

        max_dist = 0
        min_dist = float('inf')
        point_index = None

        # Calculate the distance between all pairs of points and find the maximum and minimum distances
        for point1 in original_data:
            for point2 in original_data:
                if not np.array_equal(point1, point2):
                    temp = np.linalg.norm(point1 - point2)
                    if temp > max_dist:
                        max_dist = temp
                        point_index = point1
                    min_dist = min(min_dist, temp)
        _max = 0
        for i in range(d):
            temp = point_index[i]
            _max = max(temp, _max)

        left = _max - max_dist
        right = _max + max_dist

        if left > 0:
            left_log2 = math.floor(math.log2(left))
        else:
            left_log2 = -math.ceil(math.log2(-left))

        right_log2 = math.ceil(math.log2(right))

        if left_log2 > 0:
            left_result = 2 ** left_log2
        else:
            left_result = -2 ** (-left_log2)

        right_result = 2 ** right_log2
        root_bounds = [(left_result, right_result) for _ in range(d)]

        return root_bounds

    def point_process_random(self, original_data):
        original_data = np.array(original_data)
        n, d = original_data.shape

        max_dist = 0

        max_ax = float('-inf')
        min_ax = float('inf')

        # Calculate the distance between all pairs of points and find the maximum and minimum distances
        for point1 in original_data:
            for point2 in original_data:
                if not np.array_equal(point1, point2):
                    max_dist = max(np.linalg.norm(point1 - point2), max_dist)
            for i in range(d):
                max_ax = max(max_ax, point1[i])
                min_ax = min(min_ax, point1[i])
        tmp = np.random.uniform(low=min_ax, high=max_ax)

        left = tmp - max_dist
        right = tmp + max_dist

        if left > 0:
            left_log2 = math.floor(math.log2(left))
        else:
            left_log2 = -math.ceil(math.log2(-left))

        right_log2 = math.ceil(math.log2(right))

        if left_log2 > 0:
            left_result = 2 ** left_log2
        else:
            left_result = -2 ** (-left_log2)

        right_result = 2 ** right_log2
        root_bounds = [(left_result, right_result) for _ in range(d)]

        return root_bounds

    def split_bounds(self, bounds):
        d = len(bounds)
        for i in range(2 ** d):
            new_bounds = []
            for j in range(d):
                mid = (bounds[j][0] + bounds[j][1]) / 2
                if i & (1 << j):
                    new_bounds.append((mid, bounds[j][1]))
                else:
                    new_bounds.append((bounds[j][0], mid))
            yield new_bounds

    def build_2rhst_dec(self, bounds, points, level=0, parent=None, next_id=1, max_level=0, max_id=1):
        if level == 0:
            point_ids = [i for i, p in enumerate(points) if
                         all(bounds[j][0] <= p[j] < bounds[j][1] for j in range(len(bounds)))]
        else:
            point_ids = [i for i in parent.point_ids if
                         all(bounds[j][0] <= points[i][j] < bounds[j][1] for j in range(len(bounds)))]

        node = self.Node(bounds, level, point_ids, parent, node_id=next_id)

        if parent:
            node.edge_weight = np.sqrt(len(bounds)) * abs((bounds[0][1] - bounds[0][0]))
        max_level = max(max_level, level)
        max_id = max(max_id, next_id)

        if len(point_ids) == 0:
            return None, next_id, max_level, max_id
        elif len(point_ids) == 1:
            return node, next_id, max_level, max_id

        current_id = next_id + 1

        for new_bounds in self.split_bounds(bounds):
            child, child_id, max_level, max_id = self.build_2rhst_dec(new_bounds, points, level + 1, node, current_id,
                                                                      max_level, max_id)
            if child and child.point_ids:
                node.children.append(child)
                current_id = child_id + 1
                max_id = max(max_id, current_id)

        return node, current_id, max_level, max_id

    # Correct the number of tree layers to be consistent
    def split_to_max_level(self, leaves, max_level, points, max_id):
        for node in leaves:
            if node.level < max_level:
                temp = node
                for _ in range(max_level - temp.level):
                    new_bounds_list = self.split_bounds(temp.bounds)
                    for new_bounds in new_bounds_list:
                        id = temp.id
                        child_point_ids = [i for i in temp.point_ids if all(
                            new_bounds[j][0] <= points[i][j] < new_bounds[j][1] for j in range(len(new_bounds)))]
                        if child_point_ids:
                            temp.edge_weight = np.sqrt(len(temp.bounds)) * abs((temp.bounds[0][1] - temp.bounds[0][0]))
                            child_node = self.Node(new_bounds, temp.level + 1, child_point_ids, parent=temp,
                                                   node_id=max_id + 1)
                            temp.children.append(child_node)
                            temp = child_node
                            max_id += 1
                            break
                temp.edge_weight = np.sqrt(len(temp.bounds)) * abs((temp.bounds[0][1] - temp.bounds[0][0]))

    def get_ancestors(self, node):
        #Get all ancestor nodes from the current node to the root, including the current node itself
        ancestors = []
        while node:
            ancestors.append(node)
            node = node.parent
        return ancestors

    @lru_cache
    # Sum the edge weights on the tree
    def find_shortest_path_weight(self, node1, node2):
        if node1.id == node2.id:
            return 0
        if (node1.id, node2.id) in self.dist:
            return self.dist[(node1.id, node2.id)]

        ancestors1 = self.get_ancestors(node1)
        ancestors2 = self.get_ancestors(node2)

        common_ancestor = None
        ancestors1_ids = {ancestor.id: ancestor for ancestor in ancestors1}
        ancestors2_ids = {ancestor.id: ancestor for ancestor in ancestors2}

        for ancestor_id, ancestor in ancestors1_ids.items():
            if ancestor_id in ancestors2_ids:
                common_ancestor = ancestor
                break

        weight = 0

        if common_ancestor:
            for ancestor in ancestors1:
                if ancestor == common_ancestor:
                    break
                weight += ancestor.edge_weight
            for ancestor in ancestors2:
                if ancestor == common_ancestor:
                    break
                weight += ancestor.edge_weight
        self.dist[(node1.id, node2.id)] = weight
        return weight

    def collect_leaf_nodes(self, node=None):
        if node is None:
            node = self.root
        # If there are no children, it is a leaf node
        if not node.children:
            return [node]
        else:
            leaves = []
            for child in node.children:
                if child is not None:
                    leaves += self.collect_leaf_nodes(child)
            return leaves
