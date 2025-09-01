import networkx as nx
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path('..').resolve()
sys.path.insert(0, str(project_root / 'src'))
from etc.hamiltonian import Hamiltonian


# define graph example
Gc = nx.barbell_graph(25, 5)
Hobj = Hamiltonian(Gc)
rho = Hobj.graph_density(Gc)
D2 = 1/Hobj._min_positive_Dinv2()

def alpha_params(rho_dens, D2, eta):
    return eta * (1- rho_dens)/(D2* rho_dens)

def observe_density(ratio,n_nodes):
    return ratio*n_nodes

alpha_params(rho, D2, 0.02)