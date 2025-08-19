import networkx as nx
import numpy as np

def communities_touched(G, S):
    # use fast label propagation as a rough proxy
    import networkx.algorithms.community as nxcom
    parts = list(nxcom.label_propagation_communities(G))
    label = {u:i for i,comm in enumerate(parts) for u in comm}
    return len(set(label[u] for u in S))

def mean_pairwise_distance(G, S):
    # average shortest path among S (on connected G)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(S); m=len(nodes)
    total=c=0
    for i in range(m):
        for j in range(i+1,m):
            total += dist[nodes[i]][nodes[j]]
            c += 1
    return total/c if c else 0.0