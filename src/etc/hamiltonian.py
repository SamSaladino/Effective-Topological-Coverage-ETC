import numpy as np
import networkx as nx

def precompute(G):
    """
    Precompute necessary matrices for Hamiltonian computation:
    Adjacency matrix, distance matrix, and inverse distance matrix triangularization.
    --------
    Variables:
    - A: Adjacency matrix

    Returns:
    - Dinv2_triu: Inverse square distance matrix
    - nodes: List of nodes
    - idx: Node index mapping
    """
    # Extracting nodes
    nodes = list(G.nodes())
    # Node index mapping
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    # Adjacency matrix
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.float64, format="csr")
    # Distance matrix
    dist = dict(nx.all_pairs_shortest_path_length(G))
    
    Dinv2_triu = np.zeros((n, n), dtype=np.float64)
    # Compute the inverse square of distances
    for u, du in dist.items():
        i = idx[u]
        for v, dij in du.items():
            j = idx[v]
            if i < j and dij > 0:
                Dinv2_triu[i, j] = 1.0 / (dij * dij)
    return A, Dinv2_triu, nodes, idx



def H(A, Dinv2_triu, S_idx, mu=1.0, gamma=1.0):
    """ Compute Hamiltonian of the graph for a subset of nodes
    -----------
    Variables:
    - A: Adjacency matrix
    - Dinv2_triu: Inverse square distance matrix
    - S_idx: Indices of observed nodes
    - mu: Parameter for local contributions
    - gamma: Parameter for global contributions
    -----------
    Returns:
    - H: Hamiltonian value, first term, second term
    """ 

    # Observed nodes in X experiment
    S_idx = np.asarray(S_idx, dtype=np.int64)
    n = A.shape[0]
    
    # Set nodes status for identified 1 0 for not identified
    s = np.zeros(n, dtype=np.float64)
    s[S_idx] = 1.0
    
    # Local contributions
    t1 = -0.5 * mu * float(s @ (A @ s))
    
    # Global contributions
    sub = Dinv2_triu[np.ix_(S_idx, S_idx)]
    t2 = gamma * float(np.triu(sub, k=1).sum())
    return t1 + t2, t1, t2

# -------------------- Graph Properties ------------------------


# Average shortest path length
def avg_spl(G):
    return nx.average_shortest_path_length(G)

# Density is 1 if graph is complete
def graph_density(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n <= 1:
        return 0.0
    return (2 * e) / (n * (n - 1))

# ------------------- Parameter Estimation ---------------------

def mu_density_aware(G, scale=1.0):
    rho = graph_density(G)
    return scale * max(0.0, 1.0 - rho)

def gamma_balancer(G, Dinv2_triu, mu=1.0, scale=1.0):
    """Automatically set gamma based on mu and minimum positive entry in Dinv2_triu
    """
    pos = Dinv2_triu[Dinv2_triu > 0.0]
    if pos.size > 0:
        min_pos = pos.min()
    else:
        min_pos = 0.0
    return scale * (mu * (min_pos ** 2) * graph_density(G))/(1 - graph_density(G))