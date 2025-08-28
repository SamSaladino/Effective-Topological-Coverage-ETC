import numpy as np
import networkx as nx
import pytest

from etc.hamiltonian import Hamiltonian


def test_empty_graph_behaviour():
    G = nx.Graph()
    Hobj = Hamiltonian(G)
    # precomputed matrices should be empty
    assert Hobj.A.shape == (0, 0)
    assert Hobj.Dinv2_triu.shape == (0, 0)

    # compute on empty selection should return zeros
    val, t1, t2 = Hobj.compute([], mu=1.0, gamma=None)
    assert val == 0.0 and t1 == 0.0 and t2 == 0.0


def test_single_node_graph():
    G = nx.Graph()
    G.add_node(0)
    Hobj = Hamiltonian(G)
    # shapes
    assert Hobj.A.shape == (1, 1)
    assert Hobj.Dinv2_triu.shape == (1, 1)

    # selecting the only node should yield zero contributions
    val, t1, t2 = Hobj.compute([0], mu=1.0, gamma=None)
    assert t1 == 0.0 and t2 == 0.0 and val == 0.0


def test_disconnected_graph_no_positive_dinv2():
    # two nodes, no edge
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    Hobj = Hamiltonian(G)
    # Dinv2 should be all zeros because nodes are disconnected
    assert np.all(Hobj.Dinv2_triu == 0.0)

    val, t1, t2 = Hobj.compute([0, 1], mu=1.0, gamma=None)
    assert t2 == 0.0


def test_gamma_balancer_on_complete_graph():
    # complete graph: density = 1.0 -> gamma_balancer uses fallback formula
    G = nx.complete_graph(3)
    Hobj = Hamiltonian(G)

    # min positive of Dinv2_triu should be 1.0 (distance 1 for all pairs)
    min_pos = Hobj._min_positive_Dinv2()
    assert pytest.approx(min_pos) == 1.0

    gamma = Hobj.gamma_balancer(mu=1.0, scale=1.0)
    # with rho == 1.0 the implementation returns mu * (min_pos ** 2)
    assert pytest.approx(gamma) == 1.0

    # computing on full node set: t1 = -0.5 * mu * s@(A@s)
    # For K3, s@(A@s) == 6 so t1 == -3; t2 == gamma * sum(triu(Dinv2)) == 1 * 3 == 3
    val, t1, t2 = Hobj.compute([0, 1, 2], mu=1.0, gamma=None)
    assert pytest.approx(t1) == -3.0
    assert pytest.approx(t2) == 3.0
    assert pytest.approx(val) == 0.0


def test_invalid_S_idx_raises():
    G = nx.path_graph(3)
    Hobj = Hamiltonian(G)
    # index out of range should raise an IndexError when setting s[S_idx] = 1.0
    with pytest.raises(IndexError):
        Hobj.compute([0, 5], mu=1.0, gamma=None)
