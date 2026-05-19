import numpy as np
import random
import matplotlib.pyplot as plt
from etc.hamiltonian import Hamiltonian

def frange(start: float, stop: float, step: float):
    """Yield floating-point values from start to stop (inclusive) by step.
    Parameters
    ----------
    start : float
        The starting value of the range.
    stop : float
        The stopping value of the range.
    step : float
        The step size for the range.
    Returns
    -------
    float
        The next value in the range.
    Raises
    ------
    ValueError
        If step is zero.
    """

    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0:
        while start <= stop + 1e-12:
            yield float(start)
            start += step
    else:
        while start >= stop - 1e-12:
            yield float(start)
            start += step


def build_Jij(A: np.ndarray, Div2: np.ndarray, mu: float, gamma: float):
    """ Construct the matrix with all the constant parameters in the graph
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph
    Div2 : np.ndarray
        Inverse of distance matrix power of 2
    mu : float
        Parameter mu
    gamma : float
        Parameter gamma
    Returns
    -------
    dict J       
    A dictionary where keys are tuples (i,j) and values are the 
    corresponding Jij values.
    """
    n = A.shape[0]
    J = {}
    for i in range(n):
        for j in range(i+1,n):
            Jij = (-mu*A[i,j])+gamma*Div2[i,j]*(1-A[i,j])
            if Jij != 0.0:
                J[(i,j)] = float(Jij)
    return J

def sample_k_closest_to_zero(
        H:object= Hamiltonian,
        k: int = 10,
        mu: float = None,
        gamma: float = None,
        n_random: int = 500,
        n_restarts: int = 10,
        n_local_iters: int = 200,
        swap_candidates: int = 50,
        seed: int = None,
    ):
    """
    Search for a k-node subset whose Hamiltonian value is as close to 
    zero as possible.

    Strategy:
    - Random sampling of `n_random` subsets, keep best few.
    - Local greedy swap hillclimb from top random candidates (try up 
    to n_restarts) where in each local iteration we attempt random 
    swaps of one in-subset with one out-of-subset to reduce |H|.
    - Optional exact solver fallback using `solve_extreme_k` when 
    A and D2 are provided and `use_exact=True`.

    Returns: (best_value, best_subset_list)
    """
    rng = random.Random(seed)
    n = H.n
    if not (0 < k <= n):
        raise ValueError("k must be in 1..n")

    if mu is None:
        mu = 1.0
    # if gamma None, Hamiltonian.compute will pick a default

    # helper to eval H
    def eval_H(sub):
        return float(H.compute(sub, mu=mu, gamma=gamma)[0])

    best_val = None
    best_subset = None

    # Random sampling phase
    for _ in range(n_random):
        # Sample a random subset of size k
        subset = rng.sample(range(n), k)
        # Evaluate its Hamiltonian value
        val = eval_H(subset)
        # Update best if this is the closest to zero we've seen
        if best_val is None or abs(val) < abs(best_val):
            # Update best value and subset
            best_val = val
            best_subset = list(subset)
            # If we found a perfect zero, we can stop early
            if abs(best_val) == 0.0:
                return best_val, best_subset

    # Local improvement from several restarts 
    starts = [best_subset]
    # add some random starts
    for _ in range(max(0, n_restarts - 1)):
        starts.append(rng.sample(range(n), k))
    # Local improvement phase
    for start in starts:
        cur = list(start)
        cur_set = set(cur)
        cur_val = eval_H(cur)
        improved = True
        it = 0
        # Greedy local search with random swap candidates
        while improved and it < n_local_iters:
            it += 1
            improved = False
            # sample candidate out-of-subset nodes to try swapping in
            outs = [v for v in range(n) if v not in cur_set]
            if not outs:
                break
            sampled_outs = rng.sample(
                outs, min(len(outs), swap_candidates))
            # try all possible in-subset nodes (k smallish) with 
            # sampled outs
            for u in list(cur):
                for v in sampled_outs:
                    new_subset = list(cur)
                    # swap u out, v in
                    new_subset.remove(u)
                    new_subset.append(v)
                    new_val = eval_H(new_subset)
                    # If this swap improves (reduces) the absolute 
                    # value of H, take it
                    if abs(new_val) + 1e-12 < abs(cur_val):
                        cur = new_subset
                        cur_set = set(cur)
                        cur_val = new_val
                        improved = True
                        break
                if improved:
                    break
        # check result
        if best_val is None or abs(cur_val) < abs(best_val):
            best_val = cur_val
            best_subset = list(cur)
            if abs(best_val) == 0.0:
                return best_val, best_subset

    return best_val, best_subset

def landscape_diagram_values(mu: float=1.0,
                            Hamiltonian:object=Hamiltonian, 
                            gamma:float=1.0,
                            kmax:int=10, scale_max: float=80, 
                            scale_min: float=0.25, 
                            scale_steps: float=0.25,
                            k_steps:int=1):
    """
    Compute the landscape diagram values for a given graph 
    and H function.
    
    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix of the graph
    D2 : np.ndarray
        Distance matrix squared
    mu : float
        Parameter mu
    kmax : int
        Maximum number of nodes to select
    scale_max : int
        Maximum scale for the Hamiltonian function
    Hamiltonian : object
        Hamiltonian class to use
    Returns
    -------
    k_values : np.ndarray
        Array of k values used in the landscape diagram.
    ratio : np.ndarray
        Array of mu/gamma ratios corresponding to the scales used.
    H : np.ndarray
        2D array of Hamiltonian values for each (k, scale) pair.
    """
    k_values = np.arange(2, kmax + 1, k_steps)
    scale_values = frange(
        scale_min, scale_max + scale_steps, scale_steps)

    H = np.zeros((len(k_values), len(scale_values)))
    ratio = np.zeros_like(H)
    
    for i_k, k in enumerate(k_values):
        for j_s, scale in enumerate(scale_values):
            gamma_scaled = scale * gamma
            hmin = sample_k_closest_to_zero(
                H=Hamiltonian,
                k=int(k),
                mu=mu,
                gamma=gamma_scaled,
                seed=12345,
            )[0]
            H[i_k, j_s] = hmin
            alfa = mu/gamma_scaled
            ratio[
                i_k, j_s
                ] = alfa if gamma_scaled != 0 else float("inf")

    return k_values, ratio, H

def plot_balance_landscape(
    G,
    *,
    mu=1.0,
    gamma=1.0,
    kmax=20,
    scale_max=12,
    scale_min=0.25,
    scale_steps=0.25,
    k_steps=1,
    cmap="viridis",
    contour_color="white",
    title="Balanced Configuration Accessibility Landscape",
    show_transitions=True,
    transition_levels=8,
    highlight_scale=None,
    highlight_color="red",
    highlight_linewidth=2.0,
    figures_dir : str = None,
):
    """
    Plot the balance accessibility landscape.

    Observable:
        best sampled |H|

    across:
        - k
        - gamma/gamma0 scaling
    """

  
    # Hamiltonian object
    H_obj = Hamiltonian(G)

    # Compute landscape values
    k_values, ratio, H = landscape_diagram_values(
        Hamiltonian=H_obj,
        mu=float(mu),
        gamma=float(gamma),
        kmax=int(kmax),
        scale_max=float(scale_max),
        scale_min=float(scale_min),
        scale_steps=float(scale_steps),
        k_steps=int(k_steps),
    )

    # Scale values
    scale_values = np.frange(
        scale_min,
        scale_max + scale_steps,
        scale_steps,
    )

    # Plot extent
    extent = [
        scale_values[0],
        scale_values[-1],
        k_values[0],
        k_values[-1],
    ]

    # Figure
    fig, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    # Heatmap
    im = ax.imshow(
        H,
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        cmap=cmap,
        extent=extent,
    )
    # Contours
    if show_transitions:

        h_min, h_max = np.nanmin(H), np.nanmax(H)

        contour_levels = np.linspace(
            h_min,
            h_max,
            transition_levels,
        )

        cs = ax.contour(
            scale_values,
            k_values,
            H,
            levels=contour_levels,
            colors=contour_color,
            linewidths=0.7,
            alpha=0.5,
        )

        ax.clabel(
            cs,
            inline=True,
            fontsize=7,
            fmt="%.3f",
        )

    # Highlight scaling factor
    if highlight_scale is not None:

        ax.axvline(
            x=float(highlight_scale),
            color=highlight_color,
            linestyle="--",
            linewidth=highlight_linewidth,
            alpha=0.8,
            label=rf"$\gamma/\gamma_0={highlight_scale:.2f}$",
        )

        ax.legend(loc="upper right")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)

    cbar.set_label(
        r"$H \approx 0|$"
    )

    # Labels
    ax.set_xlabel(
        r"$\gamma / \gamma_0$"
    )

    ax.set_ylabel(
        r"$k$"
    )

    ax.set_title(title)

    # Information
    print(
        f"Graph: n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}"
    )

    print("Computed balance accessibility landscape.")

    if show_transitions:

        print(
            f"Transition contours shown "
            f"at {transition_levels} levels."
        )

    if highlight_scale is not None:

        print(
            f"Highlighted scale at "
            f"{highlight_scale:.3f}"
        )

    # Save figure
    filename = (
        f"BalanceLandscape_{title}.png"
    )

    plt.savefig(
        figures_dir / filename,
        dpi=300,
    )

    plt.show()

    return fig, ax, (
        k_values,
        ratio,
        H,
    )