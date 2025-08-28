from scipy.sparse import identity
from scipy.sparse.linalg import spsolve


def compute_propagated_signal(f, L, alpha=0.1):
    """
    Compute (Id_mat + alpha * L)^(-1) f to smooth observed signal across graph.

    Parameters:
        f (np.array): binary indicator vector (1 = observed node)
        L (scipy.sparse matrix): normalized Laplacian matrix
        alpha (float): regularization strength

    Returns:
        f_prime (np.array): propagated signal vector
    """
    n = L.shape[0]
    Id_mat = identity(n, format='csr')
    A = Id_mat + alpha * L
    return spsolve(A, f)
