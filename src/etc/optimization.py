import numpy as np
import concurrent.futures
import os
from itertools import combinations
from typing import Sequence, Tuple, Optional
from etc.hamiltonian import Hamiltonian
import networkx as nx

def S0(idx,nodes):
    mask = np.zeros(len(nodes), dtype=bool)
    mask[idx] = True
    return mask


def close_nodes_sample(G:nx.Graph,k:int,seed:int=42):
    """
    Select the highest degree node and it's k-1 closest neighbors in
    the gragh. If the highest degree node has less than k-1 neighbors,
    chose the second order neighbors until k nodes are selected.
    If there are multiple nodes with the same degree, the node with the 
    lowest index is selected.

    Parameters:
    - G: nx.Graph, the input graph.
    - k: int, the number of nodes to select.
    - seed: int, the random seed for reproducibility.

    Returns:
    - sample: np.ndarray, the indices of the selected nodes.
    """
    rng = np.random.default_rng(seed)
    degrees = dict(G.degree())
    max_degree_node = max(degrees, key=degrees.get)
    neighbors = list(G.neighbors(max_degree_node))
    
    if len(neighbors) >= k - 1:
        selected_neighbors = rng.choice(neighbors, size=k-1, replace=False)
        sample = np.array([max_degree_node] + list(selected_neighbors))
    else:
        # If not enough neighbors, include second-order neighbors
        second_order_neighbors = set()
        for neighbor in neighbors:
            second_order_neighbors.update(G.neighbors(neighbor))
        second_order_neighbors.discard(max_degree_node)  # Remove the max degree node itself
        all_candidates = list(set(neighbors) | second_order_neighbors)
        
        if len(all_candidates) >= k - 1:
            selected_candidates = rng.choice(all_candidates, size=k-1, replace=False)
            sample = np.array([max_degree_node] + list(selected_candidates))
        else:
            raise ValueError("Not enough nodes to select from.")
    
    return sample

def farthest_nodes_sample(G:nx.Graph,k:int,seed:int=42,
                          distance_matrix:np.ndarray=None):
    """
    Select the k nodes that are farthest apart in the graph based on shortest path lengths.

    Parameters:
    - G: nx.Graph, the input graph.
    - k: int, the number of nodes to select.
    - seed: int, the random seed for reproducibility.

    Returns:
    - sample: np.ndarray, the indices of the selected nodes.
    """
    rng = np.random.default_rng(seed)
    if distance_matrix is None:
        distance_matrix = np.array(nx.floyd_warshall_numpy(G))  
    else:
        distance_matrix = np.array(distance_matrix)
    
    if k > len(G.nodes):
        raise ValueError("k cannot be greater than the number of nodes in the graph.")
    
    selected_nodes = []
    # Start with a random node
    first_node = rng.choice(list(G.nodes))
    selected_nodes.append(first_node)
    while len(selected_nodes) < k:
        # Calculate the maximum distance from the selected nodes to all other nodes
        max_distances = np.max([distance_matrix[selected_nodes, :]], axis=0).filled(-1)
        # Exclude already selected nodes
        max_distances[selected_nodes] = -1
        # Select the node with the maximum distance
        next_node = np.argmax(max_distances)
        selected_nodes.append(next_node)
        
    return np.array(selected_nodes)

def sampling_energy(
        n: int, 
        k: int, gamma: float, 
        mu: float, 
        n_samples: int = 10000, 
        seed: int = 42, 
        n_workers: int = 8):
    """
    Sample the energy of a Hamiltonian for a given number of samples.

    Parameters:
    - n: int, the number of elements in the system in this case nodes.
    - k: int, the number of elements to choose in each sample.
    - gamma: float, a parameter for the Hamiltonian.
    - mu: float, a parameter for the Hamiltonian.
    - n_samples: int, the number of samples to generate.
    - seed: int, the random seed for reproducibility.
    - n_workers: int, the number of parallel workers to use.

    Returns:
    - hamiltonian: np.ndarray, the sampled hamiltonian values.
    - min_hamiltonian_sample: np.ndarray, the sample with the minimum hamiltonian value.
    - max_hamiltonian_sample: np.ndarray, the sample with the maximum hamiltonian value.
    """
    n_workers = max(1, min(n_workers, n_samples))

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_workers)

    chunk_sizes = [
        n_samples // n_workers + (1 if i < n_samples % n_workers else 0)
        for i in range(n_workers)
    ]

    def worker(chunk_size, child_seed):
        rng_worker = np.random.default_rng(child_seed)
        energies_chunk = np.empty(chunk_size, dtype=float)
        samples_chunk = np.empty((chunk_size, k), dtype=int)
        for idx in range(chunk_size):
            sample = rng_worker.choice(n, size=k, replace=False)
            E = H.compute(sample, gamma=gamma, mu=mu)[0]
            energies_chunk[idx] = E
            samples_chunk[idx] = sample
        return energies_chunk, samples_chunk

    tasks = [
        (chunk_sizes[i], child_seeds[i])
        for i in range(n_workers)
        if chunk_sizes[i] > 0
    ]

    if len(tasks) == 1:
        results = [worker(*tasks[0])]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(tasks)) as executor:

            futures = [
                executor.submit(
                    worker, chunk_size, child_seed
                    ) for chunk_size, child_seed in tasks
                ]
            results = [future.result() for future in futures]

    energies = np.concatenate([r[0] for r in results])
    samples = np.concatenate([r[1] for r in results], axis=0)

    return energies, samples[energies.argmin()], samples[energies.argmax()]

if __name__ == "__main__":
    
    graph = nx.erdos_renyi_graph(n=3000, p=0.1, seed=42)

    H = Hamiltonian(graph)
    n = 100
    k = 10
    gamma = 0.5
    mu = 0.1
    n_samples = 100000
    seed = 42
    n_workers = os.cpu_count() or 1

    hamiltonian, min_hamiltonian_sample, max_hamiltonian_sample = sampling_energy(
        n, k, gamma, mu, n_samples, seed, n_workers
    )

    print(f"Minimum hamiltonian sample: {min_hamiltonian_sample}, Hamiltonian: {hamiltonian.min()}")
    print(f"Maximum hamiltonian sample: {max_hamiltonian_sample}, Hamiltonian: {hamiltonian.max()}")