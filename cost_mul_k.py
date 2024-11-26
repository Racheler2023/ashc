
import math
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import random
import time

import exponential
import greedy
import linkage
import k_means_plus
import rhst
import real_data
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


#Calculate the k-median cost function
def calculate_kmedian_cost(clusters, data):
    total_cost = 0
    for cluster in clusters:

        points = np.array([data[i] for i in cluster])

        min_cost = float('inf')

        for candidate_center in points:

            tmp = points - candidate_center
            cost = np.sum(np.linalg.norm(tmp, axis=1))

            if cost < min_cost:
                min_cost = cost

        total_cost += min_cost
    return total_cost


def calculate_MI(clusters, labels):
    # Convert clusters to an array of labels in the same format as labels.
    cluster_labels = [0] * len(labels)

    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            cluster_labels[index] = cluster_id

    ami = adjusted_mutual_info_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    return ami, nmi


"""
Run and plot the results based on different data sets
"""

start_time = time.time()

"""
Parameter settings, including:
- Data set
- Center k size
- Random algorithm run times (turn_2)
- Algorithm parameter epsilons
"""
data, labels = real_data.get_processed_wine_data()
k = 32
turn_2 = 100
epsilons = [0.1, 1, 1000]
tree = rhst.Rhst(data)
# tree.print_tree()

"""
An array storing different k-median costs. 
The array is named after the algorithm used.
"""
CLNSS_cost = []
single_cost = []
complete_cost = []
ward_cost = []
average_cost = []
k_means_plus_cost = []

our_aver_cost = []
for _ in range(len(epsilons)):
    our_aver_cost.append([0] * k)

all_cluster_1, _ = greedy.k_median_dec(tree, k)
all_cluster_5, _ = linkage.single_linkage(data, k)
all_cluster_7, _ = linkage.complete_linkage(data, k)
all_cluster_9, _ = linkage.ward_linkage(data, k)
all_cluster_11, _ = linkage.average_linkage(data, k)
all_cluster_12, _ = k_means_plus.kmeans_plus_plus_with_multiple_clusters(data, k)

for i in range(k):
    CLNSS_cost.append(calculate_kmedian_cost(all_cluster_1[i], data))
    single_cost.append(calculate_kmedian_cost(all_cluster_5[i], data))
    complete_cost.append(calculate_kmedian_cost(all_cluster_7[i], data))
    ward_cost.append(calculate_kmedian_cost(all_cluster_9[i], data))
    average_cost.append(calculate_kmedian_cost(all_cluster_11[i], data))
    k_means_plus_cost.append(calculate_kmedian_cost(all_cluster_12[i], data))

random_seeds = [random.randint(1, 2 ** 32 - 1) for _ in range(turn_2)]
eps_seed_cluster = [[] for _ in range(len(epsilons))]
eps_cost = []
for eps_index in range(len(epsilons)):
    epsilon = epsilons[eps_index]
    for j in range(turn_2):
        random_seed = random_seeds[j]
        all_cluster_3, _ = exponential.k_median_ranexp(tree, k, epsilon, random_seed)
        for index in range(k):
            cost = calculate_kmedian_cost(all_cluster_3[index], data)
            our_aver_cost[eps_index][index] += (cost / turn_2)

CLNSS_cost = CLNSS_cost[1::2]
single_cost = single_cost[1::2]
complete_cost = complete_cost[1::2]
ward_cost = ward_cost[1::2]
average_cost = average_cost[1::2]
k_means_plus_cost = k_means_plus_cost[1::2]
for i in range(len(epsilons)):
    if i < len(our_aver_cost):
        our_aver_cost[i] = our_aver_cost[i][1::2]

i_values = list(range(16))

markers = ['o', 'x', 's']
linestyles = ['--', ':', '-']

# Draw cost data for different algorithms
plt.plot(i_values, CLNSS_cost, marker='o', linestyle='-', color='orange', label='CLNSS algorithm')
plt.plot(i_values, single_cost, marker='o', linestyle='-', color='cyan', label='single linkage')
plt.plot(i_values, complete_cost, marker='o', linestyle='-', color='magenta', label='complete linkage')
plt.plot(i_values, ward_cost, marker='o', linestyle='-', color='blue', label="ward's method")
plt.plot(i_values, average_cost, marker='o', linestyle='-', color='green', label='average linkage')
plt.plot(i_values, k_means_plus_cost, marker='o', linestyle='-', color='yellow', label='K-Means+ Cost')

# Plot our_aver_cost data, handling different epsilon values
for eps_index in range(len(epsilons)):
    plt.plot(i_values, [our_aver_cost[eps_index][index] for index in range(16)],
             marker=markers[eps_index], linestyle=linestyles[eps_index], color='red',
             label=f'Average Cost (eps={epsilons[eps_index]})')

plt.xlabel('Clustering number', fontsize=20)
plt.ylabel('Cost', fontsize=20)
plt.title('Wine k-median cost', fontsize=25)

plt.xticks(ticks=i_values, labels=[str(2 * i + 2) for i in i_values], fontsize=12)
plt.legend(loc='best')

save_path = f'pic/cost/wine.png'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path)
