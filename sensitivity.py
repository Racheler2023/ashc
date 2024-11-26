import networkx as nx
from networkx.algorithms import bipartite


# Symmetric difference calculation method
def points_Similarity_Clustering(clusters1, clusters2, num):
    G = nx.Graph()

    # Treat each cluster as a node
    leftNodes = ['l' + str(i) for i in range(len(clusters1))]
    rightNodes = ['r' + str(i) for i in range(len(clusters2))]

    G.add_nodes_from(leftNodes, bipartite=0)
    G.add_nodes_from(rightNodes, bipartite=1)

    # Add edges with weights
    for i, set_A in enumerate(clusters1):
        set_A = set(set_A)
        for j, set_B in enumerate(clusters2):
            set_B = set(set_B)
            weight = -len(set_A.intersection(set_B))
            G.add_edge(leftNodes[i], rightNodes[j], weight=weight)

    # Perform matching
    matching = bipartite.matching.minimum_weight_full_matching(G, leftNodes, 'weight')

    # Calculate matching value
    value = 0
    for li in leftNodes:
        rj = matching[li]
        i = int(li[1:])
        j = int(rj[1:])
        set_A = set(clusters1[i])
        set_B = set(clusters2[j])
        intersection_size = len(set_A.intersection(set_B))
        value += intersection_size

    return num - value
