
import copy

import os

import matplotlib.pyplot as plt
import random
import time

import exponential
import greedy
import linkage
import rhst
import sensitivity
import synthetic
import real_data

# DateSet
start_time = time.time()
turn_1 = 50
turn_2 = 100
epsilons = [0.1, 1, 10, 100, 1000]

data, _ = real_data.get_processed_iris_data()
number = 16

tree = rhst.Rhst(data)
# tree.print_tree()
delete_numbers = [1, int(len(data) * 0.01), int(len(data) * 0.05), int(len(data) * 0.1)]
"""
 Delete the set of points, and there is an inheritance relationship, 
 because the latter set contains the previous set
"""
deleted_sets = [[] for _ in range(turn_1)]

"""
Each one is expressed as the size of number+1*delete_numbers,
which represents the average sensitivity on delete_index when k=i
"""
CLNSS_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
single_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
complete_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
ward_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
average_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
our_aver_sen = [[[0 for _ in range(len(delete_numbers))] for _ in range(number)] for _ in range(len(epsilons))]

# Calculate the clustering results
all_cluster_1, _ = greedy.k_median_dec(tree, number)
all_cluster_5, _ = linkage.single_linkage(data, number)
all_cluster_7, _ = linkage.complete_linkage(data, number)
all_cluster_9, _ = linkage.ward_linkage(data, number)
all_cluster_11, _ = linkage.average_linkage(data, number)

#Give the seed for the loop
random_seeds = [random.randint(1, 2 ** 10 - 1) for _ in range(turn_2)]

eps_seed_cluster = [[None for _ in range(turn_2)] for _ in range(len(epsilons))]

for i in range(len(epsilons)):
    epsilon = epsilons[i]
    for j in range(turn_2):
        random_seed = random_seeds[j]
        all_cluster_3, _ = exponential.k_median_ranexp(tree, number, epsilon, random_seed)
        # Put all_cluster_3 in the corresponding position
        eps_seed_cluster[i][j] = all_cluster_3

all_points_id = set(range(len(data)))
for del_index in range(len(delete_numbers)):
    for index_turn1 in range(turn_1):
        # Reset deleted_index at the start of each inner loop j
        delete_number = delete_numbers[del_index]
        available_points_id = copy.copy(all_points_id)
        if del_index != 0:
            delete_number -= delete_numbers[del_index - 1]
            available_points_id -= set(deleted_sets[index_turn1][del_index - 1])
        deleted_sets[index_turn1].append(random.sample(available_points_id, delete_number))
        if del_index != 0:
            deleted_sets[index_turn1][del_index] += deleted_sets[index_turn1][del_index - 1]
        deleted_index = deleted_sets[index_turn1][del_index]

        all_cluster_2, _ = greedy.k_median_dec(tree, number, deleted_index)
        all_cluster_6, _ = linkage.single_linkage(data, number, deleted_index)
        all_cluster_8, _ = linkage.complete_linkage(data, number, deleted_index)
        all_cluster_10, _ = linkage.ward_linkage(data, number, deleted_index)
        all_cluster_12, _ = linkage.average_linkage(data, number, deleted_index)

        for i in range(number):
            CLNSS_aver_sen[i][del_index] += sensitivity.points_Similarity_Clustering(all_cluster_1[i],
                                                                                     all_cluster_2[i], len(data))
            single_aver_sen[i][del_index] += sensitivity.points_Similarity_Clustering(all_cluster_5[i],
                                                                                      all_cluster_6[i], len(data))
            complete_aver_sen[i][del_index] += sensitivity.points_Similarity_Clustering(all_cluster_7[i],
                                                                                        all_cluster_8[i], len(data))
            ward_aver_sen[i][del_index] += sensitivity.points_Similarity_Clustering(all_cluster_9[i],
                                                                                    all_cluster_10[i], len(data))
            average_aver_sen[i][del_index] += sensitivity.points_Similarity_Clustering(all_cluster_11[i],
                                                                                       all_cluster_12[i], len(data))

        for i in range(len(epsilons)):
            epsilon = epsilons[i]
            for j in range(turn_2):
                random_seed = random_seeds[j]
                all_cluster_4, _ = exponential.k_median_ranexp(tree, number, epsilon, random_seed, deleted_index)
                for k in range(number):
                    our_aver_sen[i][k][del_index] += sensitivity.points_Similarity_Clustering(
                        eps_seed_cluster[i][j][k], all_cluster_4[k], len(data))

    for i in range(number):
        CLNSS_aver_sen[i][del_index] /= turn_1
        single_aver_sen[i][del_index] /= turn_1
        complete_aver_sen[i][del_index] /= turn_1

        ward_aver_sen[i][del_index] /= turn_1
        average_aver_sen[i][del_index] /= turn_1

        for j in range(len(epsilons)):
            our_aver_sen[j][i][del_index] /= (turn_1 * turn_2)

end_time = time.time()
execution_time = end_time - start_time

for j in range(number):
    fig, ax = plt.subplots()

    x_positions = range(len(delete_numbers))

    markers = ['o', 'x', 's', '^', 'd']
    lifestyles = ['--', ':', '-', '-.', '--']
    for i in range(len(epsilons)):
        plt.plot(x_positions, our_aver_sen[i][j], marker=markers[i], linestyle=lifestyles[i], color='red', label='ε=' + str(epsilons[i]))

    plt.plot(x_positions, CLNSS_aver_sen[j], marker='o', linestyle='-', color='orange', label='CLNSS algorithm')
    plt.plot(x_positions, single_aver_sen[j], marker='o', linestyle='-', color='cyan', label='single linkage')
    plt.plot(x_positions, complete_aver_sen[j], marker='o', linestyle='-', color='magenta', label='complete linkage')
    plt.plot(x_positions, ward_aver_sen[j], marker='o', linestyle='-', color='blue', label='ward\'s method')
    plt.plot(x_positions, average_aver_sen[j], marker='o', linestyle='-', color='green', label='average linkage')

    x_labels = ['1', '1%', '5%', '10%']
    plt.xticks(ticks=x_positions, labels=x_labels)

    ax.set_xlabel(r'Delete number', fontsize=20)
    ax.set_ylabel('Average Sensitivity', fontsize=20)
    ax.set_title('Iris', fontsize=25)

    ax.legend(loc='best')

    save_path = f'pic/iris/Iris centers={j+1}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
