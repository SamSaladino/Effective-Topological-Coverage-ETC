import numpy as np
import concurrent.futures
import os
from itertools import combinations
from typing import Sequence, Tuple, Optional
from etc.hamiltonian import Hamiltonian
import networkx as nx

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