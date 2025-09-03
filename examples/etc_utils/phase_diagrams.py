import networkx as nx
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import random
import multiprocessing
import numpy as np

project_root = Path('..').resolve()
sys.path.insert(0, str(project_root / 'src'))
from etc.hamiltonian import Hamiltonian


# define graph example
Gc = nx.barbell_graph(25, 5)
n = Gc.number_of_nodes()        
n_trials = 2000   
rng = random.Random(12345)
summary = {}
Hobj = Hamiltonian(Gc)
mu = Hobj.mu_density_aware(Gc)
indices = list(range(n))

for scale in range(1,45,1):
    # scale will be iterable alongside with k
    gamma = Hobj.gamma_balancer(mu=mu,scale=scale)
    
    for k in range(1,n,5):

        values = np.empty(n_trials, dtype=np.float64)
        t1_vals = np.empty(n_trials, dtype=np.float64)
        t2_vals = np.empty(n_trials, dtype=np.float64)
        subsets = [None] * n_trials

        for i in range(n_trials):
            S = rng.sample(indices, k)
            val, t1, t2 = Hobj.compute(S, mu=mu,gamma=gamma)
            values[i] = val
            t1_vals[i] = t1
            t2_vals[i] = t2
            subsets[i] = S

        imin = int(np.argmin(values))
        imax = int(np.argmax(values))

        summary['Barbell(c=25, p=5)'] = {
            "min": float(values[imin]),
            "min_idx": imin,
            "min_subset": subsets[imin],
            "max": float(values[imax]),
            "max_idx": imax,
            "max_subset": subsets[imax],
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

        print(
            f"  H min={summary['Barbell(c=25, p=5)']['min']:.6g} (trial {imin}), "
            f"max={summary['Barbell(c=25, p=5)']['max']:.6g} (trial {imax}), mean={summary['Barbell(c=25, p=5)']['mean']:.6g}"
        )



# def alpha_params(rho_dens, D2, eta):
#     return eta * (1- rho_dens)/(D2* rho_dens)

# def observe_density(ratio,n_nodes):
#     return ratio*n_nodes


# print(alpha_params(rho, D2, 0.02))
# a = np.arange(0.05,0.5,0.01)
# b = np.arange(0.1,1,0.02)

# def compute_alpha_vs_density_phase_diagram(
#     Hobj: Hamiltonian,
#     Gc: nx.Graph,
#     rho: float,
#     D2: float,
#     eta_step: float = 0.02,
#     ratio_vals: Optional[np.ndarray] = None,
#     n_trials: int = 400,
#     mu_mode: str = "1-rho",
#     use_alpha_as: str = "gamma",
# ):
#     """Compute a phase matrix of minimal H over eta (->alpha) vs observed density.

#     Parameters
#     - Hobj, Gc, rho, D2: prepared objects/values from the top of this file.
#     - eta_step: increment for eta in [0.02, 1.0].
#     - ratio_vals: 1D array of ratios to pass to observe_density; if None uses default.
#     - n_trials: Monte Carlo trials per grid cell.
#     - mu_mode: when '1-rho' set mu = 1 - rho (per your instruction). Other modes unsupported.
#     - use_alpha_as: either 'gamma' (default) or 'mu' to map alpha to that parameter.

#     Returns (eta_vals, ratio_vals, k_values, Hmin_matrix)
#     """
#     # defaults
#     if ratio_vals is None:
#         ratio_vals = np.linspace(0.05, 0.5, 24)

#     eta_vals = np.arange(0.02, 1.0 + 1e-12, eta_step)
#     n = Gc.number_of_nodes()

#     # precompute k per ratio
#     k_values = np.empty(len(ratio_vals), dtype=np.int64)
#     for j, r in enumerate(ratio_vals):
#         k = int(round(observe_density(r, n)))
#         k = max(1, min(n, k))
#         k_values[j] = k

#     indices = list(range(n))
#     rng = random.Random(12345)

#     def sample_min_H_for_k(k, mu, gamma, trials):
#         best = np.inf
#         for _ in range(trials):
#             S = rng.sample(indices, k)
#             val, _, _ = Hobj.compute(S, mu=mu, gamma=gamma)
#             if val < best:
#                 best = val
#         return float(best)

#     Hmin = np.full((len(eta_vals), len(ratio_vals)), np.nan, dtype=np.float64)

#     # Determine mu base value per instruction: mu = 1 - rho
#     if mu_mode == "1-rho":
#         mu_base = float(max(0.0, 1.0 - rho))
#     else:
#         mu_base = 1.0

#     for i, eta in enumerate(eta_vals):
#         alpha = alpha_params(rho, D2, eta)
#         for j, r in enumerate(ratio_vals):
#             k = int(k_values[j])
#             # map alpha to parameters per choice
#             if use_alpha_as == "gamma":
#                 mu = mu_base
#                 gamma = alpha
#             else:
#                 mu = alpha
#                 gamma = None

#             if not np.isfinite(gamma) and gamma is not None:
#                 Hmin[i, j] = np.nan
#                 continue

#             Hmin[i, j] = sample_min_H_for_k(k, mu, gamma, n_trials)

#     return eta_vals, ratio_vals, k_values, Hmin


# if __name__ == "__main__":
#     # quick test run (smaller n_trials for speed) and save figure
#     import os
#     import matplotlib

#     matplotlib.use("Agg")
#     out_dir = Path("../figures").resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     eta_vals, ratio_vals, k_values, Hmin = compute_alpha_vs_density_phase_diagram(
#         Hobj, Gc, rho, D2, eta_step=0.02, ratio_vals=None, n_trials=50, mu_mode="1-rho", use_alpha_as="gamma"
#     )

#     # simple plot: x axis = k (observed density), y axis = eta -> alpha
#     fig, ax = plt.subplots(figsize=(9, 6))
#     extent = [k_values[0] - 0.5, k_values[-1] + 0.5, eta_vals[0], eta_vals[-1]]
#     im = ax.imshow(Hmin, origin='lower', aspect='auto', extent=extent, cmap='seismic')
#     ax.set_xlabel('observed density (k)')
#     ax.set_ylabel('eta -> alpha')
#     ax.set_title('Minimal H (MC) across eta vs observed density (mu = 1 - rho, alpha->gamma)')
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('min H')
#     ax.set_xticks(k_values)
#     ax.set_xticklabels([str(int(k)) for k in k_values], rotation=45)
#     fig_path = out_dir / "alpha_vs_density_minH.png"
#     fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
#     print(f"Saved quick test figure to: {fig_path}")

# """
# End of serial example block removed. Below: multiprocessing-based parallel implementation
# and a main runner that uses it for a quick test.
# """


# # ---------------- Parallel version using multiprocessing -----------------
# _WORKER_STATE = {}


# def _init_worker(hobj, indices):
#     """Initializer for worker processes: store Hobj and indices in global state."""
#     global _WORKER_STATE
#     _WORKER_STATE['Hobj'] = hobj
#     _WORKER_STATE['indices'] = indices


# def _compute_cell(args):
#     """Compute minimal H for one grid cell.

#     args = (i, j, eta, alpha, k, mu, gamma, n_trials, seed)
#     Returns (i, j, best)
#     """
#     i, j, eta, alpha, k, mu, gamma, n_trials, seed = args
#     rng = random.Random(seed + i * 1009 + j * 9176)
#     Hobj = _WORKER_STATE['Hobj']
#     indices = _WORKER_STATE['indices']
#     best = float('inf')
#     for _ in range(n_trials):
#         S = rng.sample(indices, k)
#         val, _, _ = Hobj.compute(S, mu=mu, gamma=gamma)
#         if val < best:
#             best = val
#     return (i, j, best)


# def compute_alpha_vs_density_phase_diagram_parallel(
#     Hobj: Hamiltonian,
#     Gc: nx.Graph,
#     rho: float,
#     D2: float,
#     eta_step: float = 0.02,
#     ratio_vals: Optional[np.ndarray] = None,
#     n_trials: int = 400,
#     mu_mode: str = "1-rho",
#     use_alpha_as: str = "gamma",
#     n_workers: int = None,
#     seed: int = 12345,
# ):
#     """Parallel version: compute Hmin matrix using multiprocessing Pool.

#     Returns same shape results as the serial version.
#     """
#     if ratio_vals is None:
#         ratio_vals = np.linspace(0.05, 0.5, 24)

#     eta_vals = np.arange(0.02, 1.0 + 1e-12, eta_step)
#     n = Gc.number_of_nodes()

#     # precompute k per ratio
#     k_values = np.empty(len(ratio_vals), dtype=np.int64)
#     for j, r in enumerate(ratio_vals):
#         k = int(round(observe_density(r, n)))
#         k = max(1, min(n, k))
#         k_values[j] = k

#     # base mu
#     if mu_mode == "1-rho":
#         mu_base = float(max(0.0, 1.0 - rho))
#     else:
#         mu_base = 1.0

#     # prepare tasks
#     tasks = []
#     for i, eta in enumerate(eta_vals):
#         alpha = alpha_params(rho, D2, eta)
#         for j, r in enumerate(ratio_vals):
#             k = int(k_values[j])
#             if use_alpha_as == "gamma":
#                 mu = mu_base
#                 gamma = alpha
#             else:
#                 mu = alpha
#                 gamma = None

#             tasks.append((i, j, eta, alpha, k, mu, gamma, n_trials, seed))

#     Hmin = np.full((len(eta_vals), len(ratio_vals)), np.nan, dtype=np.float64)

#     if n_workers is None:
#         n_workers = max(1, multiprocessing.cpu_count() - 1)

#     indices = list(range(n))
#     # Start pool with Hobj and indices initialized in workers
#     with multiprocessing.Pool(processes=n_workers, initializer=_init_worker, initargs=(Hobj, indices)) as pool:
#         for res in pool.imap_unordered(_compute_cell, tasks, chunksize=1):
#             ii, jj, best = res
#             Hmin[ii, jj] = best

#     return eta_vals, ratio_vals, k_values, Hmin


# if __name__ == "__main__":
#     # run a quick parallel test and save figure
#     import matplotlib
#     matplotlib.use("Agg")
#     out_dir = Path("../figures").resolve()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     eta_vals, ratio_vals, k_values, Hmin = compute_alpha_vs_density_phase_diagram_parallel(
#         Hobj, Gc, rho, D2, eta_step=0.02, ratio_vals=None, n_trials=50, mu_mode="1-rho", use_alpha_as="gamma", n_workers=4
#     )

#     fig, ax = plt.subplots(figsize=(9, 6))
#     extent = [k_values[0] - 0.5, k_values[-1] + 0.5, eta_vals[0], eta_vals[-1]]
#     im = ax.imshow(Hmin, origin='lower', aspect='auto', extent=extent, cmap='seismic')
#     ax.set_xlabel('observed density (k)')
#     ax.set_ylabel('eta -> alpha')
#     ax.set_title('Minimal H (parallel MC) across eta vs observed density')
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label('min H')
#     ax.set_xticks(k_values)
#     ax.set_xticklabels([str(int(k)) for k in k_values], rotation=45)
#     fig_path = out_dir / "alpha_vs_density_minH_parallel.png"
#     fig.savefig(str(fig_path), dpi=200, bbox_inches='tight')
#     print(f"Saved parallel test figure to: {fig_path}")