import numpy as np
import networkx as nx

def precompute(G):
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.float64, format="csr")
    dist = dict(nx.all_pairs_shortest_path_length(G))
    Dinv2_triu = np.zeros((n, n), dtype=np.float64)
    for u, du in dist.items():
        i = idx[u]
        for v, dij in du.items():
            j = idx[v]
            if i < j and dij > 0:
                Dinv2_triu[i, j] = 1.0 / (dij * dij)
    return A, Dinv2_triu, nodes, idx

def H(A, Dinv2_triu, S_idx, mu=1.0, gamma=1.0):
    S_idx = np.asarray(S_idx, dtype=np.int64)
    n = A.shape[0]
    s = np.zeros(n, dtype=np.float64)
    s[S_idx] = 1.0
    t1 = -0.5 * mu * float(s @ (A @ s))
    sub = Dinv2_triu[np.ix_(S_idx, S_idx)]
    t2 = gamma * float(np.triu(sub, k=1).sum())
    return t1 + t2, t1, t2

def avg_deg(G):
    return 2.0 * G.number_of_edges() / G.number_of_nodes()

def avg_spl(G):
    return nx.average_shortest_path_length(G)

def gamma_balancer(G, k, mu=1.0):
    if k <= 1:
        return 0.0
    d_bar = avg_deg(G)
    l_bar = avg_spl(G)
    return (mu * d_bar * (l_bar ** 2)) / (k - 1)
