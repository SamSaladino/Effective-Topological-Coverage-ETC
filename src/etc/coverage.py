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

        return abs(value)
    
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

            E = self.energy(S_index, mu=mu, gamma=gamma)
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
            
            E = self.energy(S_index, mu=mu, gamma=gamma)
            energies.append(E)
            samples.append(S_index.copy())
        
        energies = np.array(energies)
        imin, imax = energies.argmin(), energies.argmax()
        return energies, samples[imin], samples[imax]
    
    
    
    # -----------------------------------------------------------#
    # Energy minimization
    # -----------------------------------------------------------#