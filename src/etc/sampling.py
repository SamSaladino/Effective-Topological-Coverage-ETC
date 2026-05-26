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

    def __init__(self, hamiltonian: Hamiltonian, seed: int = 42) -> None:
        self.H = hamiltonian
        self.seed = seed

    def _rng(self, seed: Optional[int] = None) -> np.random.Generator:
        return np.random.default_rng(self.seed if seed is None else seed)

    def _subset_to_index_set(self, subset: np.ndarray) -> set[int]:
        return set(int(idx) for idx in np.asarray(subset, dtype=np.int64))

    def _neighborhood_indices(self, node_idx: int) -> List[int]:
        node_label = self.H.nodes[int(node_idx)]
        return [self.H.idx[neighbor] for neighbor in self.H.G.neighbors(node_label)]

    def random_subset(self, k: int, seed: Optional[int] = None) -> np.ndarray:
        rng = self._rng(seed)
        return rng.choice(self.H.n, size=k, replace=False)

    def _propose_graph_swap(
        self,
        current: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """Swap one occupied node for a free neighbor of another occupied node.

        The removed node is chosen among the occupied nodes that are not adjacent
        to the selected anchor node. If no such move exists, return None.
        """

        current_set = self._subset_to_index_set(current)
        occupied = np.array(sorted(current_set), dtype=np.int64)

        if occupied.size == 0:
            return None

        for anchor_idx in rng.permutation(occupied):
            neighbors = self._neighborhood_indices(int(anchor_idx))
            free_neighbors = [idx for idx in neighbors if idx not in current_set]
            if not free_neighbors:
                continue

            non_neighbors = [
                idx for idx in occupied
                if idx != anchor_idx and idx not in neighbors
            ]
            if not non_neighbors:
                continue

            add_idx = int(rng.choice(free_neighbors))
            remove_idx = int(rng.choice(non_neighbors))

            proposal_set = set(current_set)
            proposal_set.remove(remove_idx)
            proposal_set.add(add_idx)

            return np.array(sorted(proposal_set), dtype=np.int64)

        return None

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

        The RNG is reinitialized on every call, so repeated calls with the 
        same seed return the same sample sequence, matching the older API.
        """

        rng = self._rng()

        energies: List[float] = []
        h_values: List[float] = []
        samples: List[np.ndarray] = []

        for _ in range(n_samples):
            S_idx = rng.choice(self.H.n, size=k, replace=False)
            h, _, _ = self.H.compute(S_idx, mu=mu, gamma=gamma)

            energies.append(abs(h))
            h_values.append(h)
            samples.append(S_idx.copy())

        energies = np.asarray(energies, dtype=np.float64)
        h_values = np.asarray(h_values, dtype=np.float64)
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

    def minimize_energy(
        self,
        S0: np.ndarray,
        mu: float = 1.0,
        gamma: float = 1.0,
        Tmax: float = 1.0,
        Tmin: float = 1e-6,
        cooloing: float = 0.995,
        seed: int = 42,
        steps: int = 10000,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """Simulated annealing over node-index subsets using graph-guided swaps."""

        rng = self._rng(seed)
        current = np.array(
            sorted(set(int(idx) for idx in np.asarray(S0, dtype=np.int64))),
            dtype=np.int64,
        )
        current_energy = self.H.energy(current, mu=mu, gamma=gamma)

        best_subset = current.copy()
        best_energy = current_energy
        temperature = Tmax
        history: List[float] = []

        for _ in range(steps):
            proposal = self._propose_graph_swap(current, rng)
            if proposal is None:
                history.append(current_energy)
                temperature *= cooloing
                if temperature < Tmin:
                    break
                continue

            proposal_energy = self.H.energy(proposal, mu=mu, gamma=gamma)
            delta_energy = proposal_energy - current_energy

            if delta_energy <= 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta_energy / temperature)

            if accept:
                current = proposal
                current_energy = proposal_energy
                if current_energy < best_energy:
                    best_subset = current.copy()
                    best_energy = current_energy

            history.append(current_energy)
            temperature *= cooloing
            if temperature < Tmin:
                break

        return best_subset, best_energy, history

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
        S0: np.ndarray,
        mu: float = 1.0,
        gamma: Optional[float] = None,
        n_starts: int = 10,
    ) -> Dict:
        """
        Greedy local minimization of energy using graph-guided swaps.

        Starts from the provided node-index subset and repeatedly accepts only
        improving moves that preserve subset size.
        """

        best_subset = None
        best_energy = np.inf
        rng = self._rng()

        for _ in range(n_starts):

            current = np.array(
                sorted(set(int(idx) for idx in np.asarray(S0, dtype=np.int64))),
                dtype=np.int64,
            )
            current_energy = self.H.energy(current, mu=mu, gamma=gamma)

            improved = True
            while improved:
                improved = False
                proposal = self._propose_graph_swap(current, rng)
                if proposal is None:
                    break

                proposal_energy = self.H.energy(proposal, mu=mu, gamma=gamma)
                if proposal_energy < current_energy:
                    current = proposal
                    current_energy = proposal_energy
                    improved = True

            if current_energy < best_energy:
                best_energy = current_energy
                best_subset = current.copy()

        return {"best_subset": best_subset, "best_energy": best_energy}