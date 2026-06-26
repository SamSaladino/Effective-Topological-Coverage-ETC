import networkx as nx
import numpy as np
import pytest

from etc.hamiltonian import Hamiltonian
from etc.landscape_diagrams import build_Jij, solve_extreme_k

def test_build_Jij_matches_hamiltonian_definition():
    """
    On a three-node path:

        0 -- 1 -- 2

    adjacent pairs contribute -mu, while the non-adjacent pair
    contributes gamma / d^2.
    """
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    adjacency = hamiltonian.A.toarray()
    inverse_distance_squared = hamiltonian.Dinv2_triu

    coefficients = build_Jij(
        adjacency,
        inverse_distance_squared,
        mu=2.0,
        gamma=4.0,
    )

    assert coefficients[(0, 1)] == pytest.approx(-2.0)
    assert coefficients[(1, 2)] == pytest.approx(-2.0)

    # Nodes 0 and 2 are at distance 2:
    # gamma / d² = 4 / 4 = 1.
    assert coefficients[(0, 2)] == pytest.approx(1.0)

    assert len(coefficients) == 3