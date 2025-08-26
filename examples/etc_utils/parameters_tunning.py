# Re-execute the calibration function after kernel reset
import numpy as np
import networkx as nx
from hamiltonian import precompute, H

def calibrate_gamma_for_H_positive(G, k=30, mu=None, seed=42, iters=300, tol=1e-3, gamma_init=1.0):
    """
    Calibrate gamma so that H(S) >= 0 for all sampled subsets S of size k.
    Uses binary search to find the minimum gamma such that min(H) >= 0.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    k : int
        Number of nodes in each subset
    mu : float or None
        If None, computed as (1 - density)
    seed : int
        Random seed
    iters : int
        Number of random subsets to sample
    tol : float
        Tolerance on H_min (~0)
    gamma_init : float
        Starting point for gamma scaling

    Returns
    -------
    gamma_final : float
        Calibrated gamma value ensuring H >= 0
    """
    A, Dinv2, nodes, idx = precompute(G)
    n = len(nodes)
    density = 2.0 * G.number_of_edges() / (n * (n - 1))
    if mu is None:
        mu = 1.0 - density

    rng = np.random.default_rng(seed)
    
    # Initial gamma bounds
    gamma_low = 0.0
    gamma_high = gamma_init

    # Expand gamma_high until H_min >= 0
    while True:
        Hmin = np.inf
        for _ in range(iters):
            S_idx = rng.choice(n, size=k, replace=False)
            h_val, _, _ = H(A, Dinv2, S_idx, mu=mu, gamma=gamma_high)
            if h_val < Hmin:
                Hmin = h_val
        if Hmin >= 0:
            break
        gamma_high *= 2.0

    # Binary search to find minimal gamma s.t. H_min >= 0
    while gamma_high - gamma_low > tol:
        gamma_mid = 0.5 * (gamma_low + gamma_high)
        Hmin = np.inf
        for _ in range(iters):
            S_idx = rng.choice(n, size=k, replace=False)
            h_val, _, _ = H(A, Dinv2, S_idx, mu=mu, gamma=gamma_mid)
            if h_val < Hmin:
                Hmin = h_val
        if Hmin >= 0:
            gamma_high = gamma_mid
        else:
            gamma_low = gamma_mid

    return gamma_high  # minimal gamma ensuring H >= 0 within tolerance
