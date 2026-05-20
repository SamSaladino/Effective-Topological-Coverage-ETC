import numpy as np
from itertools import combinations
from typing import Sequence, Tuple, Optional

class Coverage:
    """
    Coverage metric built on top of Hamiltonian object.
    Is a normalize value by the minimun asn maximun of Energy
    for a given k subset of nodes.
    """

    def __init__(self, hamiltonian):
        self.H = hamiltonian

    # -----------------------------------------------------------#
    # Energy definition
    # -----------------------------------------------------------#

    def energy (self,S_idx, 
                mu: float = 1.0, gamma: float = 1.0,
                ) -> float:
        """
        Compute the energy of a subset of nodes S_idx.
        Where E = |H(S_idx)|

        Returns:
        --------
        E : float
            The energy of the subset S_idx.
        """

        value, _ , _ = self.H.compute(S_idx, mu=mu, gamma=gamma)

        return abs(value),value
    
    # -----------------------------------------------------------#
    # Energy sampling
    # -----------------------------------------------------------#

    def sample_energy(self, n, k, 
                               n_samples=1000, 
                               mu: float = 1.0, gamma: float = 1.0,
                               seed=42
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random subsets of nodes and compute their energies 
        to find the distribution of energy configurations and get 
        the energy thresholds.
        
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
            # create a fresh binary mask for this sample
            S_index = np.zeros(n, dtype=int)
            nodes_idx = rng.choice(n, size=k, replace=False)
            S_index[nodes_idx] = 1

            E = self.energy(S_index, mu=mu, gamma=gamma)[0]
            energies.append(E)
            samples.append(S_index.copy())

        energies = np.array(energies)
        imin, imax = energies.argmin(), energies.argmax()
        return energies, samples[imin], samples[imax]

    def sample_mask(self, n: int, k: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate a single binary mask of length `n` with `k` ones.

        This helper makes it easy to test a single sampling step.
        """
        if rng is None:
            rng = np.random.default_rng()
        S_index = np.zeros(n, dtype=int)
        nodes_idx = rng.choice(n, size=k, replace=False)
        S_index[nodes_idx] = 1
        return S_index
    
    def sample_energy_variable_k(self, n, k_min, k_max, 
                                 n_samples=1000, 
                                 mu: float = 1.0, gamma: float = 1.0,
                                 seed=42
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            S_index = np.zeros(n, dtype=int)
            # Sample a random k value between k_min and k_max
            k = rng.integers(k_min, k_max + 1)
            
            nodes_idx = rng.choice(n, size=k, replace=False)
            
            S_index[nodes_idx] = 1
            
            E = self.energy(S_index, mu=mu, gamma=gamma)[0]
            energies.append(E)
            samples.append(S_index.copy())
        
        energies = np.array(energies)
        imin, imax = energies.argmin(), energies.argmax()
        return energies, samples[imin], samples[imax]
    
    
    
    # -----------------------------------------------------------#
    # Energy minimization
    # -----------------------------------------------------------#

    def minimize_energy(self, S0, n,
                        mu: float = 1.0, 
                        gamma: float = 1.0,
                        Tmax: float =1.0,
                        Tmin: float =1e-6,
                        cooloing : float = 0.995,
                        seed: int = 42,
                        steps: int = 10000
                        ):
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
        E_current = self.energy(S_current, mu=mu, gamma=gamma)[0]
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
                gamma=gamma
                )[0]
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