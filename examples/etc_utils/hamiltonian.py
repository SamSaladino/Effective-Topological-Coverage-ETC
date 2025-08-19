import numpy as np
import networkx as nx
from functools import lru_cache

def precompute(G):
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    dist = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    Dinv2_triu = np.zeros((n, n), dtype=np.float64)
    nodes = list(G.nodes())
    idx = {u:i for i,u in enumerate(nodes)}
    for u,du in dist.items():
        i = idx[u]
        for v,dij in du.items():
            j = idx[v]
            if i < j and dij > 0:
                Dinv2_triu[i, j] = 1.0/(dij*dij)
    return A, Dinv2_triu, nodes

def H(A, Dinv2_triu, S_idx, mu=1.0, gamma=1.0):
    s = np.zeros(A.shape[0], dtype=np.float64)
    s[S_idx] = 1.0
    t1 = -0.5 * mu * float(s @ (A @ s))                 # local attraction
    sub = Dinv2_triu[np.ix_(S_idx, S_idx)]
    t2 = gamma * float(np.triu(sub, 1).sum())           # global repulsion
    return t1 + t2, t1, t2

def avg_deg(G):
    return 2.0 * G.number_of_edges() / G.number_of_nodes()

def avg_spl(G):
    return nx.average_shortest_path_length(G)
