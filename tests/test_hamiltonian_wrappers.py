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
