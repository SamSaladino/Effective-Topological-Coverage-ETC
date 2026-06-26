from ortools.sat.python import cp_model
import numpy as np
from etc.hamiltonian import Hamiltonian

# ---------------------------------------------------------------------------
# Initial outils
# ---------------------------------------------------------------------------
def frange(start: float, stop: float, step: float):
    """Yield floating-point values from start to stop (inclusive) by step."""
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
    """
    n = A.shape[0]
    J = {}
    for i in range(n):
        for j in range(i+1,n):
            Jij = (-mu*A[i,j])+gamma*Div2[i,j]*(1-A[i,j])
            if Jij != 0.0:
                J[(i,j)] = float(Jij)
    return J






def solve_extreme_k(A: np.ndarray,
                    D: np.ndarray,
                    k: int,
                    mu: float,
                    gamma: float,
                    sense: str = "closest",
                    time_limit_s: float = None,
                    workers: int = 8,
                    precision: int = 1000):
    """
    Exact H_min / H_max for selecting exactly k nodes.
    Returns: (objective_value, x_sol: {0,1}^n)
    """
    # sense: 'min' | 'max' | 'closest' (closest to zero)
    assert sense in ("min", "max", "closest")
    n = A.shape[0]
    if not (0 < k <= n):
        raise ValueError("k must be in 1..n")

    J = build_Jij(A, D, mu, gamma)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    # y_ij only where J_ij != 0
    y = {ij: model.NewBoolVar(f"y_{ij[0]}_{ij[1]}") for ij in J.keys()}

    # cardinality: select exactly k
    model.Add(sum(x) == k)

    # McCormick linearization for y_ij = x_i * x_j
    for (i, j), yij in y.items():
        model.Add(yij <= x[i])
        model.Add(yij <= x[j])
        model.Add(yij >= x[i] + x[j] - 1)

    # linear objective or absolute-objective proxy
    if sense in ("min", "max"):
        # original floating objective (OR-Tools accepts numeric coeffs here in this project)
        objective = sum(Jij * y[(i, j)] for (i, j), Jij in J.items())
        if sense == "min":
            model.Minimize(objective)
        else:
            model.Maximize(objective)
    else:
        # minimize absolute value of the objective: we must linearize |sum(Jij * y_ij)|
        # approach: scale float Jij to integers (precision), create an IntVar for the
        # (scaled) objective, then use AddAbsEquality and minimize the absolute IntVar.
        scale = int(precision)
        # scaled integer coefficients
        scaled = {ij: int(round(Jij * scale)) for ij, Jij in J.items()}
        max_abs = sum(abs(c) for c in scaled.values())
        if max_abs == 0:
            # objective is identically zero; no need to set objective
            pass
        else:
            obj_int = model.NewIntVar(-max_abs, max_abs, "obj_int")
            # build linear expression for the scaled objective
            lin = sum(c * y[ij] for ij, c in scaled.items())
            model.Add(obj_int == lin)
            abs_obj = model.NewIntVar(0, max_abs, "abs_obj")
            model.AddAbsEquality(abs_obj, obj_int)
            model.Minimize(abs_obj)

    # solve
    solver = cp_model.CpSolver()
    if workers is not None:
        solver.parameters.num_search_workers = int(workers)
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_s)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No solution found (status=%s)" % status)

    x_sol = np.array([int(solver.Value(v)) for v in x], dtype=int)
    # compute the true (float) objective from J and solved y's because when using
    # the 'closest' mode we minimized an integer proxy (abs) and solver.ObjectiveValue()
    # will return that proxy; return the real weighted sum instead.
    obj_val = sum(Jij * int(solver.Value(y[(i, j)])) for (i, j), Jij in J.items())
    return float(obj_val), x_sol


def diagram_values(
    mu: float=1.0,Hamiltonian:object=Hamiltonian, gamma:float=1.0,
        kmax:int=10, scale_max: float=80, scale_steps: float=0.25, k_steps:int=1):
    """
    Compute the phase diagram values for a given graph 
    and Hamiltonian function.
    -------
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
    """
    k_values = np.arange(2, kmax + 1, k_steps)
    scale_values = np.arange(0.25, scale_max + scale_steps, scale_steps)

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
            ratio[i_k, j_s] = mu / gamma_scaled if gamma_scaled != 0 else float("inf")

    return k_values, ratio, H


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
    Search for a k-node subset whose Hamiltonian value is as close to zero as possible.

    Strategy:
    - Random sampling of `n_random` subsets, keep best few.
    - Local greedy swap hillclimb from top random candidates (try up to n_restarts)

    Returns: (best_value, best_subset_list)
    """
    import random
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
        subset = rng.sample(range(n), k)
        val = eval_H(subset)
        if best_val is None or abs(val) < abs(best_val):
            best_val = val
            best_subset = list(subset)
            if abs(best_val) == 0.0:
                return best_val, best_subset

    # Local improvement from several restarts (start from best random and random others)
    starts = [best_subset]
    # add some random starts
    for _ in range(max(0, n_restarts - 1)):
        starts.append(rng.sample(range(n), k))

    for start in starts:
        cur = list(start)
        cur_set = set(cur)
        cur_val = eval_H(cur)
        improved = True
        it = 0
        while improved and it < n_local_iters:
            it += 1
            improved = False
            # sample candidate out-of-subset nodes to try swapping in
            outs = [v for v in range(n) if v not in cur_set]
            if not outs:
                break
            sampled_outs = rng.sample(outs, min(len(outs), swap_candidates))
            # try all possible in-subset nodes (k smallish) with sampled outs
            for u in list(cur):
                for v in sampled_outs:
                    new_subset = list(cur)
                    # swap u out, v in
                    new_subset.remove(u)
                    new_subset.append(v)
                    new_val = eval_H(new_subset)
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


def sample_k_range_closest_to_zero(
        H: object = Hamiltonian,
        k_min: int = 1,
        k_max: int = 10,
        mu: float = None,
        gamma: float = None,
        delta: float = 0.0,
        n_random: int = 500,
        n_restarts: int = 10,
        n_local_iters: int = 200,
        swap_candidates: int = 50,
        seed: int = None,
        return_all: bool = True,
    ):
    """
    Sweep a range of k values while keeping mu and gamma fixed, and search for
    the k-node subset whose Hamiltonian value is closest to zero for each k.

    Parameters
    ----------
    H : object
        Hamiltonian instance.
    k_min : int
        Smallest k to evaluate.
    k_max : int
        Largest k to evaluate.
    mu : float
        Fixed mu parameter used for every k.
    gamma : float
        Fixed gamma parameter used for every k.
    delta : float
        Keep only results with abs(H) <= delta. Use 0.0 for exact zero only.
    n_random, n_restarts, n_local_iters, swap_candidates, seed : int
        Search controls forwarded to sample_k_closest_to_zero.
    return_all : bool
        If True, return every k result together with the filtered matches.

    Returns
    -------
    results : list[dict]
        Per-k records containing k, H, abs_H, subset, and within_delta.
    best_overall : dict | None
        Record with the smallest abs(H) across the full k range.
    filtered : list[dict]
        Only the records that satisfy abs(H) <= delta.
    """
    if k_min > k_max:
        raise ValueError("k_min must be <= k_max")
    if delta < 0:
        raise ValueError("delta must be non-negative")

    results = []
    best_overall = None
    filtered = []

    for k in range(int(k_min), int(k_max) + 1):
        best_val, best_subset = sample_k_closest_to_zero(
            H=H,
            k=k,
            mu=mu,
            gamma=gamma,
            n_random=n_random,
            n_restarts=n_restarts,
            n_local_iters=n_local_iters,
            swap_candidates=swap_candidates,
            seed=seed,
        )

        record = {
            "k": k,
            "H": float(best_val),
            "abs_H": float(abs(best_val)),
            "subset": list(best_subset),
            "within_delta": abs(best_val) <= delta,
        }
        results.append(record)

        if record["within_delta"]:
            filtered.append(record)

        if best_overall is None or record["abs_H"] < best_overall["abs_H"]:
            best_overall = record

    if return_all:
        return results, best_overall, filtered
    return filtered, best_overall


if __name__ == "__main__":
    print("phase_diagrams module imported")