import numpy as np
import networkx as nx

from etc.hamiltonian import Hamiltonian, precompute, H


def test_hamiltonian_class_vs_wrappers():
    # small path graph (nodes will be [0,1,2,3])
    G = nx.path_graph(4)

    # instantiate class and precompute via wrapper
    Hobj = Hamiltonian(G)
    A, Dinv2_triu, nodes, idx = precompute(G)

    # pick a subset using indices relative to Hobj.nodes
    S_idx = [0, 2]
    S_nodes = [Hobj.nodes[i] for i in S_idx]
    S_idx_wrapped = [idx[n] for n in S_nodes]

    # compute using class (gamma=None triggers automatic gamma)
    val_c, t1_c, t2_c = Hobj.compute(S_idx, mu=1.0, gamma=None)

    # compute using wrapper, provide same gamma used by class
    gamma = Hobj.gamma_balancer(mu=1.0)
    val_w, t1_w, t2_w = H(A, Dinv2_triu, S_idx_wrapped, mu=1.0, gamma=gamma)

    assert np.allclose([val_c, t1_c, t2_c], [val_w, t1_w, t2_w])


def test_precompute_matches_class_fields():
    G = nx.cycle_graph(5)
    Hobj = Hamiltonian(G)
    A, Dinv2_triu, nodes, idx = precompute(G)

    # shapes and simple equality checks
    assert Hobj.A.shape == A.shape
    assert Hobj.Dinv2_triu.shape == Dinv2_triu.shape
    assert Hobj.nodes == nodes
    assert Hobj.idx == idx


def test_distance_matrix_inputs_equivalent():
    # path graph where shortest paths are easy to reason about
    G = nx.path_graph(6)
    n = G.number_of_nodes()

    # baseline: default behaviour (distance_matrix=None)
    H_none = Hamiltonian(G)

    # dict-like distances as produced by NetworkX
    sp_len = dict(nx.all_pairs_shortest_path_length(G))
    H_dict = Hamiltonian(G, distance_matrix=sp_len)

    # numpy ndarray distances (shape must be (n,n))
    D_arr = np.full((n, n), np.inf, dtype=np.float64)
    for u, du in sp_len.items():
        for v, dij in du.items():
            D_arr[u, v] = float(dij)
    H_arr = Hamiltonian(G, distance_matrix=D_arr)

    # integer-indexed dict (keys are indices instead of node labels)
    sp_len_idx = {i: {j: sp_len[i][j] for j in sp_len[i].keys()} for i in range(n)}
    H_idxdict = Hamiltonian(G, distance_matrix=sp_len_idx)

    # all Dinv2_triu matrices should be identical (within tolerance)
    assert np.allclose(H_none.Dinv2_triu, H_dict.Dinv2_triu)
    assert np.allclose(H_none.Dinv2_triu, H_arr.Dinv2_triu)
    assert np.allclose(H_none.Dinv2_triu, H_idxdict.Dinv2_triu)
