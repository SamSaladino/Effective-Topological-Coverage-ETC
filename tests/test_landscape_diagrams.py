import networkx as nx
import numpy as np
import pytest

from etc.hamiltonian import Hamiltonian
from etc.landscape_diagrams import build_Jij, solve_extreme_k, sample_k_closest_to_zero

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

def test_solve_extreme_k_minimum_matches_hamiltonian():
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    adjacency = hamiltonian.A.toarray()
    inverse_distance_squared = hamiltonian.Dinv2_triu

    objective, mask = solve_extreme_k(
        A=adjacency,
        D=inverse_distance_squared,
        k=2,
        mu=2.0,
        gamma=4.0,
        sense="min",
        workers=1,
    )

    selected_indices = np.flatnonzero(mask)

    expected_h = hamiltonian.compute(
        selected_indices,
        mu=2.0,
        gamma=4.0,
    )[0]

    assert mask.shape == (3,)
    assert mask.sum() == 2
    assert objective == pytest.approx(-2.0)
    assert objective == pytest.approx(expected_h)

def test_solve_extreme_k_maximum_matches_hamiltonian():
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    adjacency = hamiltonian.A.toarray()
    inverse_distance_squared = hamiltonian.Dinv2_triu

    objective, mask = solve_extreme_k(
        A=adjacency,
        D=inverse_distance_squared,
        k=2,
        mu=2.0,
        gamma=4.0,
        sense="max",
        workers=1,
    )

    selected_indices = np.flatnonzero(mask)

    expected_h = hamiltonian.compute(
        selected_indices,
        mu=2.0,
        gamma=4.0,
    )[0]

    assert mask.sum() == 2
    assert set(selected_indices) == {0, 2}
    assert objective == pytest.approx(1.0)
    assert objective == pytest.approx(expected_h)

def test_solve_extreme_k_closest_to_zero():
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    adjacency = hamiltonian.A.toarray()
    inverse_distance_squared = hamiltonian.Dinv2_triu

    objective, mask = solve_extreme_k(
        A=adjacency,
        D=inverse_distance_squared,
        k=2,
        mu=2.0,
        gamma=4.0,
        sense="closest",
        workers=1,
        precision=10_000,
    )

    selected_indices = np.flatnonzero(mask)

    expected_h = hamiltonian.compute(
        selected_indices,
        mu=2.0,
        gamma=4.0,
    )[0]

    assert set(selected_indices) == {0, 2}
    assert objective == pytest.approx(1.0)
    assert objective == pytest.approx(expected_h)

def test_sample_k_closest_to_zero_finds_best_subset():
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    best_h, best_subset = sample_k_closest_to_zero(
        H=hamiltonian,
        k=2,
        mu=2.0,
        gamma=4.0,
        n_random=50,
        n_restarts=5,
        n_local_iters=20,
        swap_candidates=3,
        seed=42,
    )

    best_subset = np.asarray(best_subset, dtype=int)

    assert best_subset.shape == (2,)
    assert len(np.unique(best_subset)) == 2
    assert set(best_subset) == {0, 2}

    computed_h = hamiltonian.compute(
        best_subset,
        mu=2.0,
        gamma=4.0,
    )[0]

    assert best_h == pytest.approx(1.0)
    assert best_h == pytest.approx(computed_h)

def test_sample_k_closest_to_zero_local_swap_improves_solution():
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    best_h, best_subset = sample_k_closest_to_zero(
        H=hamiltonian,
        k=2,
        mu=2.0,
        gamma=4.0,
        n_random=1,
        n_restarts=1,
        n_local_iters=10,
        swap_candidates=2,
        seed=0,
    )

    assert set(best_subset) == {0, 2}
    assert best_h == pytest.approx(1.0)
    assert abs(best_h) == pytest.approx(1.0)

@pytest.mark.parametrize("invalid_k", [0, 4])
def test_sample_k_closest_to_zero_rejects_invalid_k(
    invalid_k,
):
    graph = nx.path_graph(3)
    hamiltonian = Hamiltonian(graph)

    with pytest.raises(
        ValueError,
        match="k must be in 1..n",
    ):
        sample_k_closest_to_zero(
            H=hamiltonian,
            k=invalid_k,
            mu=2.0,
            gamma=4.0,
            seed=42,
        )