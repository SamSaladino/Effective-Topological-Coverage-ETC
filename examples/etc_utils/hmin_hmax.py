from ortools.sat.python import cp_model
import numpy as np
from etc.hamiltonian import Hamiltonian

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
                    sense: str = "min",
                    time_limit_s: float = None,
                    workers: int = 8):
    """
    Exact H_min / H_max for selecting exactly k nodes.
    Returns: (objective_value, x_sol: {0,1}^n)
    """
    assert sense in ("min", "max")
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

    # linear objective
    objective = sum(Jij * y[(i, j)] for (i, j), Jij in J.items())
    if sense == "min":
        model.Minimize(objective)
    else:
        model.Maximize(objective)

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
    return float(solver.ObjectiveValue()), x_sol

def phase_diagram_values(A, D2, mu: float, kmax=10, scale_max=8, Hamiltonian: object=Hamiltonian):
    """
    Compute the phase diagram values for a given graph and Hamiltonian function.
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
    results = {}
    for k in range(2, kmax+1, 1):
        results[k] = {}
        for scale in range(1, scale_max+1, 1):
            gamma = Hamiltonian.gamma_balancer(scale=scale, mu=mu)
            hmin = solve_extreme_k(A, D2, k=k, mu=mu, gamma=gamma, sense="min")[0]
            results[k][scale] = (gamma, hmin)
    return results

if __name__ == "__main__":
    pass