import networkx as nx
import pytest
import numpy as np
from etc.hamiltonian import Hamiltonian
from etc.optimization import EnergyOptimizer

def test_sampling_energy_returns_signed_hamiltonian():
    """
    sampling_energy() returns the signed Hamiltonian H.

    Negative values indicate attraction-dominated configurations,
    while positive values indicate repulsion-dominated configurations.
    """
    graph = nx.complete_graph(4)

    hamiltonian = Hamiltonian(graph)
    optimizer = EnergyOptimizer(hamiltonian)

    h_values, minimum_sample, maximum_sample = (
        optimizer.sampling_energy(
            n=4,
            k=2,
            gamma=1.0,
            mu=1.0,
            n_samples=1,
            seed=42,
            n_workers=1,
        )
    )

    # Any selected pair in K4 is adjacent:
    # T1 = -1, T2 = 0, therefore H = -1.
    assert h_values.shape == (1,)
    assert h_values[0] == pytest.approx(-1.0)

    assert len(minimum_sample) == 2
    assert len(maximum_sample) == 2

def test_sampling_hamiltonian_is_reproducible():
    """
    Repeated calls with the same seed and worker count must return
    identical Hamiltonian values and extrema samples.
    """
    graph = nx.path_graph(8)

    hamiltonian = Hamiltonian(graph)
    optimizer = EnergyOptimizer(hamiltonian)

    arguments = {
        "n": 8,
        "k": 3,
        "gamma": 1.0,
        "mu": 1.0,
        "n_samples": 100,
        "seed": 42,
        "n_workers": 2,
    }

    result_1 = optimizer.sampling_energy(**arguments)
    result_2 = optimizer.sampling_energy(**arguments)

    assert np.array_equal(result_1[0], result_2[0])
    assert np.array_equal(result_1[1], result_2[1])
    assert np.array_equal(result_1[2], result_2[2])

def test_minimum_annealing_energy_matches_returned_mask():
    """
    The energy returned by annealing must correspond to the nodes
    selected in the returned binary mask.

    A complete graph is used because every subset of k nodes has a
    simple exact Hamiltonian:

        H = -C(k, 2)

    when mu = 1 and T2 excludes adjacent pairs.
    """
    graph = nx.complete_graph(6)

    hamiltonian = Hamiltonian(graph)
    optimizer = EnergyOptimizer(hamiltonian)

    # Select three nodes using a binary mask.
    initial_mask = np.zeros(6, dtype=int)
    initial_mask[[0, 2, 4]] = 1

    best_mask, best_energy, history = (
        optimizer.min_energy_annealing(
            initial_mask,
            mu=1.0,
            gamma=1.0,
            steps=10,
            n_workers=1,
            seed=42,
        )
    )

    # The swap operation must preserve k.
    assert best_mask.sum() == 3

    # Convert the returned mask to the node positions expected by
    # Hamiltonian.compute().
    selected_positions = np.flatnonzero(best_mask)

    expected_energy = abs(
        hamiltonian.compute(
            selected_positions,
            mu=1.0,
            gamma=1.0,
        )[0]
    )

    # In K6, any three selected nodes form three edges:
    # H = -3 and E = |H| = 3.
    assert expected_energy == pytest.approx(3.0)

    # The optimizer must report the energy of the returned configuration.
    assert best_energy == pytest.approx(expected_energy)

    assert len(history) > 0