import numpy as np
from typing import Optional, Dict, List, Tuple
from itertools import combinations
from etc.hamiltonian import Hamiltonian


class Sampler:
    """
    Sampling and optimization methods for exploring the Hamiltonian
    energy landscape.

    This class does NOT define the Hamiltonian itself. It operates
    on an existing Hamiltonian object.
    """

    def __init__(self, hamiltonian: Hamiltonian, seed: Optional[int] = 42) -> None:
        self.H = hamiltonian
        self.seed = seed

    # ========================================================================
    # Fixed-k random sampling
    # ========================================================================

    def sample_fixed_k(
        self,
        k: int,
        n_samples: int = 1000,
        mu: float = 1.0,
        gamma: float = 1.0,
    ) -> Dict:
        """Sample random subsets of nodes and compute their energies.

        The RNG is reinitialized on every call, so repeated calls with the same
        seed return the same sample sequence, matching the older API.
        """

        rng = np.random.default_rng(self.seed)

        energies = np.empty(n_samples, dtype=np.float64)
        h_values = np.empty(n_samples, dtype=np.float64)
        samples: List[np.ndarray] = []

        for i in range(n_samples):
            S_idx = rng.choice(self.H.n, size=k, replace=False)
            h, _, _ = self.H.compute(S_idx, mu=mu, gamma=gamma)

            energies[i] = abs(h)
            h_values[i] = h
            samples.append(S_idx.copy())

        imin = energies.argmin()
        imax = energies.argmax()

        return {
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

    # =========================================================================
    # Variable-k random sampling
    # =========================================================================

    def sample_variable_k(
        self,
        k_min: int,
        k_max: int,
        n_samples: int = 1000,
        mu: float = 1.0,
        gamma: float = 1.0,
    ) -> Dict:
        """Sample random subsets with variable subset size."""

        if k_min < 0:
            raise ValueError("k_min must be >= 0")

        if k_max > self.H.n:
            raise ValueError("k_max exceeds graph size")

        if k_min > k_max:
            raise ValueError("k_min must be <= k_max")

        rng = np.random.default_rng(self.seed)

        energies = np.empty(n_samples, dtype=np.float64)
        h_values = np.empty(n_samples, dtype=np.float64)
        k_values = np.empty(n_samples, dtype=np.int64)
        samples: List[np.ndarray] = []

        for i in range(n_samples):
            k = rng.integers(k_min, k_max + 1)
            S_idx = rng.choice(self.H.n, size=k, replace=False)
            h, _, _ = self.H.compute(S_idx, mu=mu, gamma=gamma)

            energies[i] = abs(h)
            h_values[i] = h
            k_values[i] = k
            samples.append(S_idx.copy())

        imin = energies.argmin()
        imax = energies.argmax()

        return {
            "energies": energies,
            "hamiltonians": h_values,
            "samples": samples,
            "min_subset": samples[imin],
            "max_subset": samples[imax],
            "min_energy": energies[imin],
            "max_energy": energies[imax],
        }

    # --------------------  OPTIMIZATION ---------------------

    # ========================================================================
    # Annealing
    # ========================================================================

    def minimize_energy(
        self,
        S0,
        mu: float = 1.0,
        gamma: float = 1.0,
        Tmax: float = 1.0,
        Tmin: float = 1e-6,
        cooloing: float = 0.995,
        seed: int = 42,
        steps: int = 10000,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Minimize E using simulated anneling.
        S0 is the initial subset of nodes closest to the minimum energy configuration.
        S0[i] = 1 if node i is in the subset, 0 otherwise.

        Constraints:
        sum(S0) = k is preserved during the optimization.

        Returns:
        --------
        S_min : np.ndarray
            The subset of nodes with the minimum energy.
        E_min : float
            The minimum energy value.
        """
        rng = np.random.default_rng(seed)

        # initial configuration
        S_current = S0.copy()

        # compute initial energy
        E_current = self.energy(S_current, mu=mu, gamma=gamma)
        k = S_current.sum()

        best_S = S_current.copy()
        best_E = E_current

        # Temperature
        T = Tmax
        history = []

        # Anneling loop
        for step in range(steps):

            proposal_S = S_current.copy()
            #Find occupied and unoccupied indices
            occupied_indices = np.where(proposal_S == 1)[0]
            unoccupied_indices = np.where(proposal_S == 0)[0]

            # Swap move: randomly select one occupied and one unoccupied
            # to swap their states

            remove_node = rng.choice(occupied_indices)
            add_node = rng.choice(unoccupied_indices)

            proposal_S[remove_node] = 0
            proposal_S[add_node] = 1

            # Compute new energy
            proposal_E = self.energy(
                proposal_S,
                mu=mu,
                gamma=gamma,
            )
            delta_E = proposal_E - E_current

            # Metropolis criterion
            if delta_E < 0:
                accept = True
            else:
                prob = np.exp(-delta_E / T)
                accept = rng.random() < prob

            # Accept move
            if accept:
                S_current = proposal_S
                E_current = proposal_E

                # Update best solution
                if E_current < best_E:
                    best_S = S_current.copy()
                    best_E = E_current

            # Store history
            history.append((E_current))

            # Cool down
            T *= cooloing
            if T < Tmin:
                break

            return best_S, best_E, history

    # ========================================================================
    # Exhaustive search (small graphs only)
    # ========================================================================

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

        best_subset = None
        worst_subset = None

        best_energy = np.inf
        worst_energy = -np.inf

        for subset in combinations(range(self.H.n), k):

            subset = np.array(subset)

            h, _, _ = self.H.compute(
                subset,
                mu=mu,
                gamma=gamma,
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

    # ========================================================================
    # Greedy minimization
    # ========================================================================

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
                    gamma=gamma,
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
                            gamma=gamma,
                        )

                        proposal_E = abs(proposal_h)

                        if proposal_E < current_E:

                            current = proposal
                            current_E = proposal_E

                            improved = True

                if current_E < best_energy:

                    best_energy = current_E
                    best_subset = current.copy()

        return {"best_subset": best_subset, "best_energy": best_energy}