import numpy as np
import networkx as nx
from scipy import sparse
from typing import Sequence, Optional, Tuple, List, Dict
from .hamiltonian import Hamiltonian

def Energy (
    G: nx.Graph,
    S_idx: Sequence[int],
    mu: float = 1.0,
    gamma: float = 1.0,
    distance_matrix: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Compute the energy of a subset of nodes in a graph.

    Args:
        G: Input graph (networkx Graph).
        S_idx: Indices of the subset of nodes to evaluate.
        mu: Weight for the coverage term.
        gamma: Weight for the diversity term.
        distance_matrix: Optional precomputed distance matrix (n x n).

    Returns:
        A tuple containing:
            - Energy value (float)
            - Coverage term (float)
            - Diversity term (float)
    """
    E = abs(Hamiltonian(G, distance_matrix))
    return E.compute(S_idx, mu=mu, gamma=gamma)