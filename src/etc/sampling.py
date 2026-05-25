import numpy as np
from typing import Optional, Dict, List, Tuple

from .hamiltonian import Hamiltonian


class Sampler:
    """
    Sampling and optimization methods for exploring
    the Hamiltonian energy landscape.

    This class does NOT define the Hamiltonian itself.
    It operates on an existing Hamiltonian object.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        seed: Optional[int] = 42,
    ) -> None:

        self.H = hamiltonian
        self.rng = np.random.default_rng(seed)

    # ============================================================
    # Core random subset generation
    # ============================================================

    def random_subset(self, k: int) -> np.ndarray:
        """
        Generate a random subset of node indices.

        Parameters
        ----------
        k : int
            Number of nodes in subset.

        Returns
        -------
        np.ndarray
            Random subset of node indices.
        """

        if k < 0 or k > self.H.n:
            raise ValueError("k must satisfy 0 <= k <= number of nodes")

        return self.rng.choice(
            self.H.n,
            size=k,
            replace=False
        )

    # ============================================================
    # Fixed-k random sampling
    # ============================================================

    def sample_fixed_k(
        self,
        k: int,
        n_samples: int = 1000,
        mu: float = 1.0,
        gamma: Optional[float] = None,
    ) -> Dict:
        """
        Sample random subsets of fixed size k.

        Parameters
        ----------
        k : int
            Subset size.

        n_samples : int
            Number of random samples.

        mu : float
            Local attraction parameter.

        gamma : Optional[float]
            Global repulsion parameter.

        Returns
        -------
        Dict
            Dictionary containing energies,
            Hamiltonian values, subsets,
            and extrema information.
        """

        energies = np.empty(n_samples, dtype=np.float64)
        h_values = np.empty(n_samples, dtype=np.float64)

        samples: List[np.ndarray] = []

        for i in range(n_samples):

            S_idx = self.random_subset(k)

            h, _, _ = self.H.compute(
                S_idx,
                mu=mu,
                gamma=gamma
            )

            energies[i] = abs(h)
            h_values[i] = h

            samples.append(S_idx.copy())

        imin = energies.argmin()
        imax = energies.argmax()

        return {
            "k": k,
            "energies": energies,
            "hamiltonians": h_values,
            "samples": samples,
            "min_subset": samples[imin],
            "max_subset": samples[imax],
            "min_energy": energies[imin],
            "max_energy": energies[imax],
            "min_hamiltonian": h_values[imin],
            "max_hamiltonian": h_values[imax],
        }

    # ============================================================
    # Variable-k random sampling
    # ============================================================

    def sample_variable_k(
        self,
        k_min: int,
        k_max: int,
        n_samples: int = 1000,
        mu: float = 1.0,
        gamma: Optional[float] = None,
    ) -> Dict:
        """
        Sample random subsets with variable subset size.

        Parameters
        ----------
        k_min : int
            Minimum subset size.

        k_max : int
            Maximum subset size.

        n_samples : int
            Number of random samples.

        Returns
        -------
        Dict
            Dictionary containing sampled k values,
            energies, Hamiltonians, and subsets.
        """

        if k_min < 0:
            raise ValueError("k_min must be >= 0")

        if k_max > self.H.n:
            raise ValueError("k_max exceeds graph size")

        if k_min > k_max:
            raise ValueError("k_min must be <= k_max")

        energies = np.empty(n_samples, dtype=np.float64)
        h_values = np.empty(n_samples, dtype=np.float64)
        k_values = np.empty(n_samples, dtype=np.int64)

        samples: List[np.ndarray] = []

        for i in range(n_samples):

            k = self.rng.integers(
                k_min,
                k_max + 1
            )

            S_idx = self.random_subset(k)

            h, _, _ = self.H.compute(
                S_idx,
                mu=mu,
                gamma=gamma
            )

            energies[i] = abs(h)
            h_values[i] = h
            k_values[i] = k

            samples.append(S_idx.copy())

        imin = energies.argmin()
        imax = energies.argmax()

        return {
            "k_values": k_values,
            "energies": energies,
            "hamiltonians": h_values,
            "samples": samples,
            "min_subset": samples[imin],
            "max_subset": samples[imax],
            "min_energy": energies[imin],
            "max_energy": energies[imax],
        }

    # ============================================================
    # Exhaustive search (small graphs only)
    # ============================================================

    def exhaustive_search_k(
        self,
        k: int,
        mu: float = 1.0,
        gamma: Optional[float] = None,
    ) -> Dict:
        """
        Exhaustive search over all subsets of size k.

        WARNING:
        Computationally explosive.
        Only feasible for small graphs.

        Returns
        -------
        Dict
            Exact extrema information.
        """

        from itertools import combinations

        best_subset = None
        worst_subset = None

        best_energy = np.inf
        worst_energy = -np.inf

        for subset in combinations(range(self.H.n), k):

            subset = np.array(subset)

            h, _, _ = self.H.compute(
                subset,
                mu=mu,
                gamma=gamma
            )

            E = abs(h)

            if E < best_energy:
                best_energy = E
                best_subset = subset.copy()

            if E > worst_energy:
                worst_energy = E
                worst_subset = subset.copy()

        return {
            "best_subset": best_subset,
            "worst_subset": worst_subset,
            "best_energy": best_energy,
            "worst_energy": worst_energy,
        }

    # ============================================================
    # Greedy minimization
    # ============================================================

    def greedy_minimize(
        self,
        k: int,
        mu: float = 1.0,
        gamma: Optional[float] = None,
        n_starts: int = 10,
    ) -> Dict:
        """
        Greedy local minimization of energy.

        Starts from random subsets and iteratively
        improves the configuration.
        """

        best_subset = None
        best_energy = np.inf

        for _ in range(n_starts):

            current = self.random_subset(k)

            improved = True

            while improved:

                improved = False

                current_h, _, _ = self.H.compute(
                    current,
                    mu=mu,
                    gamma=gamma
                )

                current_E = abs(current_h)

                for remove_idx in range(k):

                    for add_node in range(self.H.n):

                        if add_node in current:
                            continue

                        proposal = current.copy()
                        proposal[remove_idx] = add_node

                        proposal_h, _, _ = self.H.compute(
                            proposal,
                            mu=mu,
                            gamma=gamma
                        )

                        proposal_E = abs(proposal_h)

                        if proposal_E < current_E:

                            current = proposal
                            current_E = proposal_E

                            improved = True

                if current_E < best_energy:

                    best_energy = current_E
                    best_subset = current.copy()

        return {
            "best_subset": best_subset,
            "best_energy": best_energy,
        }

