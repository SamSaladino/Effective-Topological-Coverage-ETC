from matplotlib.pylab import seed
import numpy as np
import networkx as nx
from scipy import sparse
from typing import Sequence, Optional, Tuple, List, Dict

class Hamiltonian:
    """Hamiltonian computation for a graph.

    Usage:
        H = Hamiltonian(G)
        value, t1, t2 = H.compute(S_idx, mu=1.0, gamma=1.0)
        For the vector S_idx, use node indices relative to the graph G.

    Note:The class preserves the prior behavior but groups related data and helpers.
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
            # fill D with inf initially
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
        - IndexError if all indices are binary (0 or 1) but not valid node indices.
        """
        # Validate S_idx input
        if not isinstance(S_idx, (list, tuple, np.ndarray)):
            raise TypeError("S_idx must be a sequence of integers")
        # Check if S_idx looks like a binary mask but contains no valid indices
        if all(isinstance(
            x, (int, np.integer)) and (x == 0 or x == 1) for x in S_idx
            ):
            raise IndexError(
                "S_idx appears to be a binary mask but contains no valid node indices"
                )

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
#========================================================================================
    # -------------------- Energy and sampling ---------------------
    def energy (
            self,S_idx: Sequence[int], mu: float = 1.0,  
            gamma: Optional[float] = None
            ) -> float:
        """Compute the energy of a subset of nodes S_idx.
        Where E = |H(S_idx)|
        Returns:
        --------
        E : float
            The energy of the subset S_idx.
        """
        H, _, _ = self.compute(S_idx, mu=mu, gamma=gamma)
        return abs(H)

    def sampling_energy(
            self,k:int, n_samples=1000,
            mu: float = 1.0, gamma: float = 1.0,
            seed=42, module: bool = True,
            return_raw_h: bool = False
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """Sample random subsets of nodes and compute their energies
        to find the distribution of energy configurations and get
        the energy thresholds.
        
        Parameters:
        - k: size of subsets to sample
        - n_samples: number of random subsets to sample
        - mu: local contribution multiplier
        - gamma: global contribution multiplier. If None, computed via 
        ``gamma_balancer``.
        - seed: random seed for reproducibility
        - module: if True, return absolute energy values; if False,
        return signed energies
        - return_raw_h: if True, second return value contains signed
        Hamiltonian values. If False (default), second return value uses
        the same convention as `module`.
        Returns:
        --------
        energies : np.ndarray
            Array of energy values for the sampled subsets.
        h_values : np.ndarray
            Array of sampled Hamiltonian values. By default this matches
            `energies`; set `return_raw_h=True` for signed values.
        min_energy_subset : np.ndarray
            The subset of nodes with the minimum energy.
        max_energy_subset : np.ndarray
            The subset of nodes with the maximum energy.
            """
        if k < 0 or k > self.n:
            raise ValueError("k must satisfy 0 <= k <= number of graph nodes")
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")

        rng = np.random.default_rng(seed)
        energies = []
        samples = []
        h_values = []
        for _ in range(n_samples):
            # vector of nodes index for _ sample
            S_index = rng.choice(self.n, size=k, replace=False)

            h = self.compute(S_index, mu=mu, gamma=gamma)[0]
            E = abs(h) if module else h
            energies.append(E)
            h_values.append(h if return_raw_h else E)
            # store the sample subset
            samples.append(S_index.copy())

        energies = np.array(energies)
        h_values = np.array(h_values)
        imin, imax = energies.argmin(), energies.argmax()

        return energies,h_values, samples[imin], samples[imax] 
    
    def sample_energy_variable_k(self, k_min, k_max, 
                                 n_samples=1000, 
                                 mu: float = 1.0, gamma: float = 1.0,
                                 seed=42,
                                 ) -> np.ndarray:
        """
        Sample random subsets of nodes with variable cardinality and compute their energies 
        to find the distribution of energy configurations and get 
        the energy thresholds.
        
        The number of nodes in each sample varies between k_min and k_max.
        
        Returns:
        --------
        energies : np.ndarray
            Array of energy values for the sampled subsets.
        min_energy_subset : np.ndarray
            The subset of nodes with the minimum energy.
        max_energy_subset : np.ndarray
            The subset of nodes with the maximum energy.
        """
    
        rng = np.random.default_rng(seed)
        energies = []
        samples = []
    
    
        for _ in range(n_samples):
            
            # Sample a random k value between k_min and k_max
            k = rng.integers(k_min, k_max + 1)

            S_index = rng.choice(self.n, size=k, replace=False)

            E = self.energy(S_index, mu=mu, gamma=gamma)
            energies.append(E)
            samples.append(S_index.copy())
        
        energies = np.array(energies)
        return energies
 #==========================================================================================   
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
        # If there are no positive entries, or min_pos is zero/invalid, return 0
        if not np.isfinite(min_pos) or min_pos <= 0.0:
            return 0.0
        rho = Hamiltonian.graph_density(self.G)
        if rho >= 1.0:
            # fallback: avoid division by zero when rho == 1.0
            return scale * mu * 1.0 / (min_pos)
        denom = min_pos * (1.0 - rho)
        if denom == 0.0:
            return 0.0
        return (scale * (mu * rho)) / denom


# ----------------- Backwards compatible thin wrappers --------------
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
