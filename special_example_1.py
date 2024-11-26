
# for linkage

import math
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time

import exponential
import greedy
import linkage
import rhst
import sensitivity
import synthetic


def random_unit_vector(d):
    """Generate a random unit vector of dimension d"""
    vector = np.random.normal(0, 1, d)
    magnitude = np.linalg.norm(vector)
    return vector / magnitude


def create_points_on_line_random_direction(n, d):
    direction = random_unit_vector(d)
    points = np.zeros((n, d))
    points[0, :] = 0
    for i in range(1, n):

        points[i, :] = points[i - 1, :] + (1 + i / n) * direction

    return points


matplotlib.use('TkAgg') # You can choose 'Agg', 'Qt5Agg', 'TkAgg', 'GTK3Agg', 'MacOSX', etc. according to your system environment


array_n = [50, 100, 150, 200, 250, 300]
our_aver_sen = []
linkage_aver_sen = []

for index_n in range(len(array_n)):

    n = array_n[index_n]

    d = 8
    data = create_points_on_line_random_direction(n, d)
    number = 16
    epsilon = 1
    turn_1 = 20
    turn_2 = 10

    tree = rhst.Rhst(data)
    tree.print_tree()

    all_point_ran_distance = [[] for _ in range(number)]
    all_point_linkage_distance = [[] for _ in range(number)]
    start_time = time.time()
    for _ in range(turn_1):
        deleted_index = set()
        deleted_index = set(random.sample(range(len(data)), 1))

        all_cluster_5, all_costs5 = linkage.single_linkage(data, number)
        all_cluster_6, all_costs6 = linkage.single_linkage(data, number, deleted_index)


        all_value = [0]
        for index_k in range(1,number):
            cluster_5 = all_cluster_5[index_k]
            cluster_6 = all_cluster_6[index_k]
            all_value.append(sensitivity.points_Similarity_Clustering(cluster_5, cluster_6, len(data)))
        for index_k in range(1,number):
            all_point_linkage_distance[index_k].append(all_value[index_k])
            print(f"Average_sensitivity of single linkage:{all_value[index_k]}")

        all_value1 = [0]
        for index_k in range(1,number):
            all_value1.append(0)
        for _ in range(turn_2):
            random_seed = random.randint(0, 2**32 - 1)
            all_cluster_3, all_costs3 = exponential.k_median_ranexp(tree, number, epsilon, random_seed)
            all_cluster_4, all_costs4 = exponential.k_median_ranexp(tree, number, epsilon, random_seed, deleted_index)
            for index_k in range(1,number):
                cluster_3 = all_cluster_3[index_k]
                cluster_4 = all_cluster_4[index_k]
                all_value1[index_k] += sensitivity.points_Similarity_Clustering(cluster_3, cluster_4, len(data))
        for index_k in range(1,number):
            all_point_ran_distance[index_k].append(math.ceil(all_value1[index_k]/turn_2))
            print(f"Average_sensitivity ran:{math.ceil(all_value1[index_k]/turn_2)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Algorithm execution time: {execution_time} seconds")

    our_aver_sen.append(0)
    linkage_aver_sen.append(0)
    for i in range(turn_1):
        our_aver_sen[len(our_aver_sen)-1] += all_point_ran_distance[index_k][i]
        linkage_aver_sen[len(linkage_aver_sen)-1] += all_point_linkage_distance[index_k][i]
    our_aver_sen[len(our_aver_sen) - 1] /= turn_1
    linkage_aver_sen[len(linkage_aver_sen)-1] /= turn_1

# Create a figure and an axes
fig, ax = plt.subplots()

# Draw the first line on the axes
ax.plot(array_n, our_aver_sen, label='Our algorithm', marker='o', color='blue') # Blue with dot markers

# Draw the second line on the axes
ax.plot(array_n, linkage_aver_sen, label='Single linkage', marker='s', color='green') # Green with square markers

# Set the legend
ax.legend()

# Add a title and axis labels
ax.set_title('Single linkage case')
ax.set_xlabel('Data size')
ax.set_ylabel('Average sensitivity')

# Show the figure
plt.show()