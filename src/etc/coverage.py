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
                mu: float = 1.0, gamma: float = 1.0
                ) -> float:
        """
        Compute the energy of a subset of nodes S_idx.
        Where E = |H(S_idx)|

        Returns:
        --------
        E : float
            The energy of the subset S_idx.
        """

        value, t1, t2 = self.H.compute(S_idx, mu=mu, gamma=gamma)

        return abs(value)
    
    # -----------------------------------------------------------#
    # Energy minimization
    # -----------------------------------------------------------#