import networkx as nx
import pytest

from etc.hamiltonian import Hamiltonian
from etc.optimization import EnergyOptimizer

def test_sampling_energy_returns_absolute_hamiltonian():
    """
    sampling_energy() should return E = |H|, not the signed
    Hamiltonian value.
    """
    graph = nx.complete_graph(4)

    hamiltonian = Hamiltonian(graph)
    optimizer = EnergyOptimizer(hamiltonian)

    energies, minimum_sample, maximum_sample = (
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

    # Any pair in K4 is connected:
    # H = T1 + T2 = -1 + 0 = -1
    # E = |H| = 1
    assert energies.shape == (1,)
    assert energies[0] == pytest.approx(1.0)

    assert len(minimum_sample) == 2
    assert len(maximum_sample) == 2