
import math
from collections import Counter
from itertools import islice

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time

from sklearn.preprocessing import StandardScaler

import exponential
import greedy
import linkage
import rhst
import sensitivity
from sklearn.datasets import make_blobs, make_regression
# You can choose 'Agg', 'Qt5Agg', 'TkAgg', 'GTK3Agg', 'MacOSX', etc. according to your system environment
matplotlib.use('TkAgg')

start_time = time.time()
n = 500

d = 3
number = 16
data, labels = make_regression(n_samples=n, n_features=d, noise=0.1, random_state=2322323)
scaler = StandardScaler()
data = scaler.fit_transform(data)
epsilons = [0.1, 1, 1000]
turn_1 = 50
turn_2 = 10

tree = rhst.Rhst(data)
tree.print_tree()

our_aver_sen = [[0 for _ in range(number-1)] for _ in epsilons]
random_seeds = [random.randint(1, 2**32 - 1) for _ in range(turn_2)]
eps_seed_clusters = []
for i in range(len(epsilons)):
    eps_seed_clusters.append([])
    epsilon = epsilons[i]
    for j in range(turn_2):
        random_seed = random_seeds[j]
        all_cluster_3, all_costs3 = exponential.k_median_ranexp(tree, number, epsilon, random_seed)
        eps_seed_clusters[i].append(all_cluster_3)

deleted_ids = []
for index_turn1 in range(turn_1):
    deleted_ids.append(set(random.sample(range(len(data)), 1)))

for index_turn1 in range(turn_1):
    deleted_index = deleted_ids[index_turn1]

    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        values = [0 for _ in range(number)]
        for j in range(turn_2):
            random_seed = random_seeds[j]
            all_cluster_4, all_costs4 = exponential.k_median_ranexp(tree, number, epsilon, random_seed, deleted_index)
            for index_k in range(1, number):
                values[index_k] += sensitivity.points_Similarity_Clustering(eps_seed_clusters[i][j][index_k], all_cluster_4[index_k], len(data))
        for index_k in range(1, number):
            our_aver_sen[i][index_k-1] += values[index_k]/turn_2
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Algorithm execution time: {execution_time} seconds")
for i in range(len(epsilons)):
    for index_k in range(0, number-1):
        our_aver_sen[i][index_k] /= turn_1
# Create a plot
fig, ax = plt.subplots()

# Generate equally spaced values
x_positions = range(2, number+1)

colors = ['green', 'red', 'blue']
for i in range(len(epsilons)):
    plt.plot(x_positions, our_aver_sen[i], marker='o', linestyle='-', color=colors[i], label='ε='+str(epsilons[i]))

# Set labels
ax.set_xlabel(r'Data size', fontsize=20)
ax.set_ylabel('Average Sensitivity', fontsize=20)

# Add legend
ax.legend(loc='best')

# Display graph
plt.show()