import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import matplotlib.pyplot as plt
import random
import time
import os
from tqdm import tqdm

import linkage
import exponential
import greedy
import rhst
import sensitivity
import real_data

def process_inner_loop(deleted_index, del_index, rhst_tree, number, data, all_cluster_1, all_cluster_5, all_cluster_7, 
                       all_cluster_9, all_cluster_11, epsilons, seeds_chunks, eps_seed_cluster):
       
    # Calculate clustering results
    all_cluster_2, _ = greedy.k_median_dec(rhst_tree, number, deleted_index)
    all_cluster_6, _ = linkage.single_linkage(data, number, deleted_index)
    all_cluster_8, _ = linkage.complete_linkage(data, number, deleted_index)
    all_cluster_10, _ = linkage.ward_linkage(data, number, deleted_index)
    all_cluster_12, _ = linkage.average_linkage(data, number, deleted_index)

    # Calculate sensitivity
    CLNSS_sen = [0] * number
    single_sen = [0] * number
    complete_sen = [0] * number
    ward_sen = [0] * number
    average_sen = [0] * number
    length=len(data)
    for i in range(number):
        CLNSS_sen[i] = sensitivity.points_Similarity_Clustering(all_cluster_1[i], all_cluster_2[i], length)
        single_sen[i] = sensitivity.points_Similarity_Clustering(all_cluster_5[i], all_cluster_6[i], length)
        complete_sen[i] = sensitivity.points_Similarity_Clustering(all_cluster_7[i], all_cluster_8[i], length)
        ward_sen[i] = sensitivity.points_Similarity_Clustering(all_cluster_9[i], all_cluster_10[i], length)
        average_sen[i] = sensitivity.points_Similarity_Clustering(all_cluster_11[i], all_cluster_12[i], length)

    # Parallel tasks section
    our_sen = [[0 for _ in range(number)] for _ in range(len(epsilons))]
    with ProcessPoolExecutor() as executor:
        futures = []
        for eps_index in range(len(epsilons)):
            for chunk_index, seeds_chunk in enumerate(seeds_chunks):
                future = executor.submit(
                    parallel_task_for_epsilon,
                    eps_index, epsilons[eps_index], seeds_chunk, chunk_index, rhst_tree, number,
                    deleted_index, eps_seed_cluster[eps_index], len(data), del_index
                )
                futures.append(future)
                
        for future in as_completed(futures):
            eps_index, del_index, chunk_index, results = future.result()
            start = chunk_index * chunk_size
            end = start + len(results)
            for i in range(start, end):
                for j in range(number):
                    our_sen[eps_index][j] += results[i - start][j]

    return CLNSS_sen, single_sen, complete_sen, ward_sen, average_sen, our_sen

def parallel_task_for_epsilon_exp(eps_index, eps, seeds_chunk, chunk_index, rhst_tree, k):
    results = []
    for i in range(len(seeds_chunk)):
        try:
            seed = seeds_chunk[i]
            clusters, _ = exponential.k_median_ranexp(rhst_tree, k, eps, seed)
            results.append(clusters)
        except Exception as e:
            print(f"Error processing epsilon {eps}, seed {seed}, iteration {i}: {str(e)}")
            continue
    return eps_index, chunk_index, results

def parallel_task_for_epsilon(eps_index, eps, seeds_chunk, chunk_index, rhst_tree, k, deleted_set, clusters, data_len,
                               del_index):
    results = []
    for i in range(len(seeds_chunk)):
        index=chunk_index*len(seeds_chunk)+i
        try:
            seed = seeds_chunk[i]
            all_cluster_4, _ = exponential.k_median_ranexp(rhst_tree, k, eps, seed, deleted_set)
            cluster_sensitivities = []
            for j in range(k):
                try:
                    value = sensitivity.points_Similarity_Clustering(clusters[index][j], all_cluster_4[j], data_len)
                    cluster_sensitivities.append(value)
                except Exception as e:
                    print(f"Error calculating sensitivity for epsilon {eps}, seed {seed}, cluster {j}, iteration {i}: {str(e)}")
                    print("Error details:", e)
                    continue  # Continue to the next cluster
            results.append(cluster_sensitivities)
        except Exception as e:
            print(f"Error processing epsilon {eps}, seed {seed}, iteration {i}: {str(e)}")
            print("Error details:", e)
            continue  # Continue to the next iteration
    return eps_index, del_index, chunk_index, results


if __name__ == '__main__':

    start_time = time.time()
    turn_1 = 100
    turn_2 = 40
    epsilons = [0.1,1,1000]

    data, _ = real_data.get_processed_iris_data()
    number = 32
    tree = rhst.Rhst(data)

    delete_numbers = [1, int(len(data) * 0.01), int(len(data) * 0.05), int(len(data) * 0.1)]
    deleted_sets = [[] for _ in range(turn_1)]

    CLNSS_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
    single_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
    complete_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
    ward_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]
    average_aver_sen = [[0 for _ in range(len(delete_numbers))] for _ in range(number)]

    all_cluster_1, _ = greedy.k_median_dec(tree, number)
    all_cluster_5, _ = linkage.single_linkage(data, number)
    all_cluster_7, _ = linkage.complete_linkage(data, number)
    all_cluster_9, _ = linkage.ward_linkage(data, number)
    all_cluster_11, _ = linkage.average_linkage(data, number)

    random_seeds = [random.randint(1, 2 ** 32 - 1) for _ in range(turn_2)] 
    chunk_size = turn_2 // 10
    seeds_chunks = [random_seeds[i:i + chunk_size] for i in range(0, turn_2, chunk_size)]  
    manager = multiprocessing.Manager()
    eps_seed_cluster = manager.list([manager.list([manager.list([0] * number) for _ in range(turn_2)]) for _ in range(len(epsilons))])
    our_aver_sen = manager.list([manager.list([manager.list([0] * len(delete_numbers)) for _ in range(number)]) for _ in range(len(epsilons))])

    with ProcessPoolExecutor() as executor:
        futures = []
        for eps_index in range(len(epsilons)):
            for chunk_index, seeds_chunk in enumerate(seeds_chunks):  
                future = executor.submit(parallel_task_for_epsilon_exp, eps_index, epsilons[eps_index], seeds_chunk, chunk_index, tree, number)
                futures.append(future)

        for future in as_completed(futures):
            eps_index, chunk_index, results = future.result()
            start = chunk_index * chunk_size
            end = start + len(results) 
            for i in range(start, end):
                eps_seed_cluster[eps_index][i] = results[i - start]
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time after the algorithm executes the first parallelization: {execution_time} seconds")

    all_points_id = set(range(len(data)))

    deleted_sets = []
    for index_turn1 in range(turn_1):
        current_deleted_set = []
        available_points_id = set(range(len(data)))  
        for del_index in range(len(delete_numbers)):
            delete_number = delete_numbers[del_index]
            if del_index != 0:
                delete_number -= delete_numbers[del_index - 1]
                available_points_id -= set(current_deleted_set[del_index - 1])
            # Convert a collection to a list
            available_points_list = list(available_points_id)
            # Sampling a list using random.sample
            deleted_points = random.sample(available_points_list, delete_number)
            current_deleted_set.append(deleted_points)
            if del_index != 0:
                current_deleted_set[del_index] += current_deleted_set[del_index - 1]
        deleted_sets.append(current_deleted_set)
        
        
    # Main loop
    for del_index in tqdm(range(len(delete_numbers)), desc="Outer loop progress"):
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for index_turn1 in range(turn_1):
                future = executor.submit(
                    process_inner_loop,
                    deleted_sets[index_turn1][del_index],del_index, tree, number, data,
                    all_cluster_1, all_cluster_5, all_cluster_7, all_cluster_9, all_cluster_11,
                    epsilons, seeds_chunks, eps_seed_cluster
                )
                futures.append(future)

            # Wait for all tasks to complete and collect results
            CLNSS_results = []
            single_results = []
            complete_results = []
            ward_results = []
            average_results = []
            our_results = []
            for future in as_completed(futures):
                CLNSS_sen, single_sen, complete_sen, ward_sen, average_sen, our_sen = future.result()
                CLNSS_results.append(CLNSS_sen)
                single_results.append(single_sen)
                complete_results.append(complete_sen)
                ward_results.append(ward_sen)
                average_results.append(average_sen)
                our_results.append(our_sen)

            # Summarize the results
            for i in range(number):
                CLNSS_aver_sen[i][del_index] = sum(result[i] for result in CLNSS_results) / turn_1
                single_aver_sen[i][del_index] = sum(result[i] for result in single_results) / turn_1
                complete_aver_sen[i][del_index] = sum(result[i] for result in complete_results) / turn_1
                ward_aver_sen[i][del_index] = sum(result[i] for result in ward_results) / turn_1
                average_aver_sen[i][del_index] = sum(result[i] for result in average_results) / turn_1
            for j in range(len(epsilons)):
                for i in range(number):
                    our_aver_sen[j][i][del_index] = sum(result[j][i] for result in our_results) / (turn_1 * turn_2)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Algorithm execution time: {execution_time} seconds + {our_aver_sen}")
    
for j in range(number):
    # Create a plot
    fig, ax = plt.subplots()

    # Generate equally spaced values ​​on the x-axis
    x_positions = range(len(delete_numbers))

    # Create a line chart
    markers = ['o', 'x', 's', '^', 'd']  
    linestyles = ['--', ':', '-', '-.', '--']  
    for i in range(len(epsilons)):
        plt.plot(x_positions, our_aver_sen[i][j], marker=markers[i], linestyle=linestyles[i], color='red', label='ε='+str(epsilons[i]))

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

    save_path = f'pic/iris/iris-{j+1}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

