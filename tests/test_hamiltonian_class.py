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

    # For K3, all three node pairs are directly connected.
    #
    # Therefore:
    # - T1 contains three attractive edge contributions: T1 = -3.
    # - T2 is zero because the factor (1 - A_ij) excludes all
    #   directly connected pairs.
    # - The total Hamiltonian is H = T1 + T2 = -3.
    val, t1, t2 = Hobj.compute(
        [0, 1, 2],
        mu=1.0,
        gamma=None,
    )

    assert t1 == pytest.approx(-3.0)
    assert t2 == pytest.approx(0.0)
    assert val == pytest.approx(-3.0)


def test_invalid_S_idx_raises():
    G = nx.path_graph(3)
    Hobj = Hamiltonian(G)
    # index out of range should raise an IndexError when setting s[S_idx] = 1.0
    with pytest.raises(IndexError):
        Hobj.compute([0, 5], mu=1.0, gamma=None)


def test_compute_is_order_invariant():
    G = nx.path_graph(5)
    Hobj = Hamiltonian(G)



    val_sorted, t1_sorted, t2_sorted = Hobj.compute([0, 2, 4], mu=1.0, gamma=None)
    val_unsorted, t1_unsorted, t2_unsorted = Hobj.compute([4, 0, 2], mu=1.0, gamma=None)

    assert pytest.approx(val_sorted) == val_unsorted
    assert pytest.approx(t1_sorted) == t1_unsorted
    assert pytest.approx(t2_sorted) == t2_unsorted


def test_adjacent_pair_contributes_only_to_local_term():
    graph = nx.path_graph(2)
    hamiltonian = Hamiltonian(graph)

    H_value, T1, T2 = hamiltonian.compute(
        [0, 1],
        mu=2.0,
        gamma=3.0,
    )

    assert T1 == pytest.approx(-2.0)
    assert T2 == pytest.approx(0.0)
    assert H_value == pytest.approx(-2.0)

def test_t2_excludes_adjacent_nodes_and_includes_non_adjacent_nodes():
    """
    In a three-node path:

        0 -- 1 -- 2

    nodes 0 and 1 are adjacent, so they must not contribute to T2.

    Nodes 0 and 2 are non-adjacent and have distance 2, so their
    contribution to T2 is gamma / 2^2.
    """
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    mu = 2.0
    gamma = 4.0

    # Adjacent pair: only T1 contributes.
    H_adjacent, T1_adjacent, T2_adjacent = hamiltonian.compute(
        [0, 1],
        mu=mu,
        gamma=gamma,
    )

    assert T1_adjacent == pytest.approx(-2.0)
    assert T2_adjacent == pytest.approx(0.0)
    assert H_adjacent == pytest.approx(-2.0)

    # Non-adjacent pair: only T2 contributes.
    # Distance(0, 2) = 2, therefore gamma / d^2 = 4 / 4 = 1.
    H_non_adjacent, T1_non_adjacent, T2_non_adjacent = (
        hamiltonian.compute(
            [0, 2],
            mu=mu,
            gamma=gamma,
        )
    )

    assert T1_non_adjacent == pytest.approx(0.0)
    assert T2_non_adjacent == pytest.approx(1.0)
    assert H_non_adjacent == pytest.approx(1.0)