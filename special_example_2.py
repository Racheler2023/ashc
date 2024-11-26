
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


def create_special_example_2( n = 301,  d = 2):
    points = np.zeros((n, d))

    points[0, :] = 0
    points[1, 0] = 15.5

    tmp = int((n-1)/3)
    points[2:tmp+1, :] = 5
    points[tmp+1:2*tmp+1, :] = 9
    points[2*tmp+1:3*tmp+1, 0] = 9
    points[2*tmp+1:3*tmp+1, 1] = 3
    # Random perturbations
    noise = np.random.normal(0, 0.01, points.shape)
    noise[:2, :] = 0
    points += noise

    return points


array_n = [52, 100, 151, 202, 250, 301]
CLNSS_aver_sen = []
our_aver_sen = []

for index_n in range(len(array_n)):
    n = array_n[index_n]
    d = 2
    data = create_special_example_2(n, d)
    number = 2
    epsilon = 1
    turn_1 = 20
    turn_2 = 10

    tree = rhst.Rhst(data)
    tree.print_tree()

    all_point_distance = [[] for _ in range(number)]
    all_point_ran_distance = [[] for _ in range(number)]
    start_time = time.time()
    for _ in range(turn_1):
        deleted_index = set()
        deleted_index = set(random.sample(range(len(data)), 1))

        all_cluster_1, all_costs1 = greedy.k_median_dec(tree, number)
        all_cluster_2, all_costs2 = greedy.k_median_dec(tree, number, deleted_index)

        all_value = [0]
        for index_k in range(1, number):
            cluster_1 = all_cluster_1[index_k]
            cluster_2 = all_cluster_2[index_k]
            all_value.append(sensitivity.points_Similarity_Clustering(cluster_1, cluster_2, len(data)))
        for index_k in range(1, number):
            all_point_distance[index_k].append(all_value[index_k])
            print(f"Average_sensitivity of single linkage:{all_value[index_k]}")

        all_value1 = [0]
        for index_k in range(1, number):
            all_value1.append(0)
        for _ in range(turn_2):
            random_seed = random.randint(0, 2**32 - 1)
            all_cluster_3, all_costs3 = exponential.k_median_ranexp(tree, number, epsilon, random_seed)
            all_cluster_4, all_costs4 = exponential.k_median_ranexp(tree, number, epsilon, random_seed, deleted_index)
            for index_k in range(1, number):
                cluster_3 = all_cluster_3[index_k]
                cluster_4 = all_cluster_4[index_k]
                all_value1[index_k] += sensitivity.points_Similarity_Clustering(cluster_3, cluster_4, len(data))
        for index_k in range(1, number):
            all_point_ran_distance[index_k].append(math.ceil(all_value1[index_k]/turn_2))
            print(f"Average_sensitivity ran:{math.ceil(all_value1[index_k]/turn_2)}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Algorithm execution time: {execution_time} seconds")

    CLNSS_aver_sen.append(0)
    our_aver_sen.append(0)
    for i in range(turn_1):
        CLNSS_aver_sen[len(CLNSS_aver_sen) - 1] += all_point_distance[index_k][i]
        our_aver_sen[len(our_aver_sen) - 1] += all_point_ran_distance[index_k][i]
    CLNSS_aver_sen[len(CLNSS_aver_sen) - 1] /= turn_1
    our_aver_sen[len(our_aver_sen) - 1] /= turn_1

matplotlib.use('TkAgg')
# Create a figure and an axes
fig, ax = plt.subplots()

# Draw the first line on the axes
ax.plot(array_n, our_aver_sen, label='Our algorithm', marker='o', color='blue') # Blue with dot markers

# Draw the second line on the axes
ax.plot(array_n, CLNSS_aver_sen, label='CLNSS algorithm', marker='s', color='green') # Green with square markers

# Set the legend
ax.legend()

# Add a title and axis labels
ax.set_title('CLNSS algorithm case')
ax.set_xlabel('Data size')
ax.set_ylabel('Average sensitivity')

# Show the figure
plt.show()
