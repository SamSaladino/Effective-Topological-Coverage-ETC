import numpy as np
import networkx as nx
from scipy import sparse
from typing import Sequence, Optional, Tuple, List, Dict

class Hamiltonian:
    """Hamiltonian computation for a graph.

    Usage:
        H = Hamiltonian(G)
        value, t1, t2 = H.compute(S_idx, mu=1.0, gamma=1.0)

    The class preserves the prior behavior but groups related data and helpers.
    """

    def __init__(self, G: nx.Graph, distance_matrix: np.ndarray = None) -> None:
        self.G = G
        self.nodes: List = list(G.nodes())
        self.idx: Dict = {u: i for i, u in enumerate(self.nodes)}
        self.n = len(self.nodes)

        # adjacency as scipy sparse csr matrix
        if self.n == 0:
            # empty CSR when graph has no nodes
            self.A = sparse.csr_matrix((0, 0), dtype=np.float64)
            # empty distance and Dinv2 matrices
            self.distance_matrix = np.zeros((0, 0), dtype=np.float64)
            self.Dinv2_triu = np.zeros((0, 0), dtype=np.float64)
            return

        self.A = nx.to_scipy_sparse_array(
            G, nodelist=self.nodes, dtype=np.float64, format="csr"
        )

        # Normalize distance_matrix input to a (n,n) numpy array named D
        D = None
        if distance_matrix is None:
            # compute shortest path lengths (dict of dicts) and fill numpy array
            sp_len = dict(nx.all_pairs_shortest_path_length(G))
            D = np.full((self.n, self.n), np.inf, dtype=np.float64)
            for u, du in sp_len.items():
                if u not in self.idx:
                    continue
                i = self.idx[u]
                for v, dij in du.items():
                    if v not in self.idx:
                        continue
                    j = self.idx[v]
                    D[i, j] = float(dij)
        else:
            # distance_matrix provided by caller: accept ndarray or dict-like
            if isinstance(distance_matrix, np.ndarray):
                D = np.asarray(distance_matrix, dtype=np.float64)
                if D.shape != (self.n, self.n):
                    raise ValueError(
                        "distance_matrix ndarray must have shape (n, n) " \
                        "matching graph nodes"
                    )
            elif isinstance(distance_matrix, dict):
                D = np.full((self.n, self.n), np.inf, dtype=np.float64)
                # support either node-keyed dicts or integer-indexed dicts
                for u, du in distance_matrix.items():
                    try:
                        i = self.idx[u]
                    except Exception:
                        # try treating u as integer index
                        try:
                            i = int(u)
                        except Exception:
                            continue
                        if i < 0 or i >= self.n:
                            continue
                    for v, dij in du.items():
                        try:
                            j = self.idx[v]
                        except Exception:
                            try:
                                j = int(v)
                            except Exception:
                                continue
                        if j < 0 or j >= self.n:
                            continue
                        D[i, j] = float(dij)
            else:
                raise TypeError(
                    "distance_matrix must be None, a numpy.ndarray, or a dict mapping"
                )

        # store the normalized distance matrix and compute Dinv2_triu
        self.distance_matrix = D
        self.Dinv2_triu = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dij = D[i, j]
                if np.isfinite(dij) and dij > 0:
                    self.Dinv2_triu[i, j] = 1.0 / (dij * dij)

    # -------------------- Core computation ---------------------
    def _min_positive_Dinv2(self, eps: float = 0.0) -> float:
        """Return minimum strictly positive entry of Dinv2_triu or eps if none.

        eps is returned when no positive entries exist (default 0.0).
        """
        pos = self.Dinv2_triu[self.Dinv2_triu > 0.0]
        if pos.size > 0:
            return float(pos.min())
        return float(eps)

    def compute(
        self, S_idx: Sequence[int], mu: float = 1.0, gamma: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Compute Hamiltonian for subset S_idx.

        Parameters
        - S_idx: sequence of integer node indices (0-based, relative to this
          Hamiltonian instance)
        - mu: local contribution multiplier
        - gamma: global contribution multiplier. If None, computed via
          ``gamma_balancer``.

        Returns a tuple (H, t1, t2).

        Raises
        - TypeError if S_idx is not a sequence of integers.
        - IndexError if any index in S_idx is out of range [0, n).
        """
        if not isinstance(S_idx, (list, tuple, np.ndarray)):
            raise TypeError("S_idx must be a sequence of integers")

        S_idx_arr = np.asarray(S_idx, dtype=np.int64)
        n = self.A.shape[0]

        if S_idx_arr.size > 0:
            if S_idx_arr.min() < 0 or S_idx_arr.max() >= n:
                raise IndexError("S_idx contains index outside the range" " of nodes")

        s = np.zeros(n, dtype=np.float64)
        s[S_idx_arr] = 1.0

        t1 = -0.5 * mu * float(s @ (self.A @ s))

        if gamma is None:
            gamma = self.gamma_balancer(mu=mu)

        sub = self.Dinv2_triu[np.ix_(S_idx_arr, S_idx_arr)]
        t2 = gamma * float(np.triu(sub, k=1).sum())
        return t1 + t2, t1, t2

    # ---------------- Graph properties & parameter estimation -----
    @staticmethod
    def graph_density(G: nx.Graph) -> float:
        n = G.number_of_nodes()
        e = G.number_of_edges()
        if n <= 1:
            return 0.0
        return (2 * e) / (n * (n - 1))

    @staticmethod
    def mu_density_aware(G: nx.Graph, scale: float = 1.0) -> float:
        rho = Hamiltonian.graph_density(G)
        return scale * max(0.0, 1.0 - rho)

    def gamma_balancer(self, mu: float = 1.0, scale: float = 1.0) -> float:
        """Automatically set gamma based on mu and minimum positive entry.

        If the graph density is 1.0, uses a fallback formula to avoid division
        by zero.
        """
        min_pos = self._min_positive_Dinv2()
        rho = Hamiltonian.graph_density(self.G)
        if rho >= 1.0:
            return scale * mu * (min_pos ** 2)
        return (scale * (mu * (min_pos ** 2) * rho)) / (1.0 - rho)


# ----------------- Backwards compatible thin wrappers -----------------
def precompute(G: nx.Graph) -> Tuple:
    """Compatibility wrapper that returns (A, Dinv2_triu, nodes, idx).

    NOTE: prefer using the ``Hamiltonian`` class directly.
    """
    H = Hamiltonian(G)
    return H.A, H.Dinv2_triu, H.nodes, H.idx


def H(
    A: sparse.spmatrix,
    Dinv2_triu: np.ndarray,
    S_idx: Sequence[int],
    mu: float = 1.0,
    gamma: float = 1.0,
    ) -> Tuple[float, float, float]:
    
    """Compatibility wrapper matching the original function signature.

    Parameters are the adjacency ``A``, the upper-triangular inverse-square
    distance matrix ``Dinv2_triu``, and ``S_idx`` a sequence of integer node
    indices.
    """
    if not isinstance(S_idx, (list, tuple, np.ndarray)):
        raise TypeError("S_idx must be a sequence of integers")

    S_idx_arr = np.asarray(S_idx, dtype=np.int64)
    n = A.shape[0]
    if S_idx_arr.size > 0:
        if S_idx_arr.min() < 0 or S_idx_arr.max() >= n:
            raise IndexError("S_idx contains index outside the range of nodes")

    s = np.zeros(n, dtype=np.float64)
    s[S_idx_arr] = 1.0
    t1 = -0.5 * mu * float(s @ (A @ s))
    sub = Dinv2_triu[np.ix_(S_idx_arr, S_idx_arr)]
    t2 = gamma * float(np.triu(sub, k=1).sum())
    return t1 + t2, t1, t2
