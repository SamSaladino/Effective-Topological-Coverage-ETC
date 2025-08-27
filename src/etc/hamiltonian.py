import numpy as np
import networkx as nx


class Hamiltonian:
    """Encapsulate Hamiltonian precomputation and computation for a graph.

    Usage:
        H = Hamiltonian(G)
        value, t1, t2 = H.compute(S_idx, mu=1.0, gamma=1.0)

    The class preserves the prior behavior but groups related data and helpers.
    """

    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.idx = {u: i for i, u in enumerate(self.nodes)}
        self.n = len(self.nodes)
        # adjacency as scipy sparse csr matrix
        self.A = nx.to_scipy_sparse_array(G, nodelist=self.nodes, dtype=np.float64, format="csr")
        # precompute inverse-square upper triangular distance matrix
        self.Dinv2_triu = np.zeros((self.n, self.n), dtype=np.float64)
        dist = dict(nx.all_pairs_shortest_path_length(G))
        for u, du in dist.items():
            i = self.idx[u]
            for v, dij in du.items():
                j = self.idx[v]
                if i < j and dij > 0:
                    self.Dinv2_triu[i, j] = 1.0 / (dij * dij)

    # -------------------- Core computation ---------------------
    def _min_positive_Dinv2(self, eps=0.0):
        """Return minimum strictly positive entry of Dinv2_triu or eps if none."""
        pos = self.Dinv2_triu[self.Dinv2_triu > 0.0]
        if pos.size > 0:
            return float(pos.min())
        return float(eps)

    def compute(self, S_idx, mu=1.0, gamma=None):
        """Compute Hamiltonian for subset S_idx.

        If gamma is None it's set to self.gamma_balancer(mu).
        Returns: (H, t1, t2)
        """
        S_idx = np.asarray(S_idx, dtype=np.int64)
        n = self.A.shape[0]
        s = np.zeros(n, dtype=np.float64)
        s[S_idx] = 1.0

        t1 = -0.5 * mu * float(s @ (self.A @ s))

        if gamma is None:
            gamma = self.gamma_balancer(mu=mu)

        sub = self.Dinv2_triu[np.ix_(S_idx, S_idx)]
        t2 = gamma * float(np.triu(sub, k=1).sum())
        return t1 + t2, t1, t2

    # ---------------- Graph properties & parameter estimation -----
    @staticmethod
    def avg_spl(G):
        return nx.average_shortest_path_length(G)

    @staticmethod
    def graph_density(G):
        n = G.number_of_nodes()
        e = G.number_of_edges()
        if n <= 1:
            return 0.0
        return (2 * e) / (n * (n - 1))

    @staticmethod
    def mu_density_aware(G, scale=1.0):
        rho = Hamiltonian.graph_density(G)
        return scale * max(0.0, 1.0 - rho)

    def gamma_balancer(self, mu=1.0, scale=1.0):
        """Automatically set gamma based on mu and minimum positive entry in self.Dinv2_triu."""
        min_pos = self._min_positive_Dinv2()
        rho = Hamiltonian.graph_density(self.G)
        # avoid division by zero when graph is complete (rho == 1.0)
        if rho >= 1.0:
            return scale * mu * (min_pos ** 2)
        return scale * (mu * (min_pos ** 2) * rho) / (1.0 - rho)


# ----------------- Backwards compatible thin wrappers -----------------
def precompute(G):
    """Compatibility wrapper that returns A, Dinv2_triu, nodes, idx like before."""
    H = Hamiltonian(G)
    return H.A, H.Dinv2_triu, H.nodes, H.idx


def H(A, Dinv2_triu, S_idx, mu=1.0, gamma=1.0):
    """Compatibility wrapper matching the original function signature.

    This keeps the old API: H(A, Dinv2_triu, S_idx, mu=1.0, gamma=1.0)
    """
    S_idx = np.asarray(S_idx, dtype=np.int64)
    n = A.shape[0]
    s = np.zeros(n, dtype=np.float64)
    s[S_idx] = 1.0
    t1 = -0.5 * mu * float(s @ (A @ s))
    sub = Dinv2_triu[np.ix_(S_idx, S_idx)]
    t2 = gamma * float(np.triu(sub, k=1).sum())
    return t1 + t2, t1, t2
