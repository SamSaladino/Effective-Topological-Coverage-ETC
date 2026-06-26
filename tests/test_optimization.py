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