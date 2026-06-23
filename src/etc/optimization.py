import numpy as np
from itertools import combinations
from typing import Sequence, Tuple, Optional

def sample_energy(self, n, k, 
                               n_samples=1000, 
                               mu: float = 1.0, gamma: float = 1.0,
                               seed=42,
                               module: bool = True
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

            E = self.energy(S_index, mu=mu, gamma=gamma, module=module)
            energies.append(E)
            samples.append(S_index.copy())

        energies = np.array(energies)
        imin, imax = energies.argmin(), energies.argmax()
        return energies, samples[imin], samples[imax]