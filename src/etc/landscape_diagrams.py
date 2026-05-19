import networkx as nx
import matplotlib.pyplot as plt
import sys
from pathlib import Path

project_root = Path('..').resolve()
sys.path.insert(0, str(project_root / 'src'))
from etc.hamiltonian import Hamiltonian

def alpha_params(mu=1, gamma=1):
    return mu/gamma

def observe_density(k_nodes,n_nodes):
    return k_nodes/n_nodes