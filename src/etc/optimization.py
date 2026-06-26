import numpy as np
import concurrent.futures
import pickle
import os
from etc.hamiltonian import Hamiltonian
import networkx as nx


class EnergyOptimizer:
    """
    A class for energy optimization and sampling operations on Graphs
    and Hamiltonian energy evaluation.
    """
    
    def __init__(self, hamiltonian: Hamiltonian):
        """
        Initialize the EnergyOptimizer with a Hamiltonian.
        
        Parameters:
        -----------
        hamiltonian : Hamiltonian
            The Hamiltonian object for the system.
        """
        self.H = hamiltonian
    
    @staticmethod
    def create_S0_mask(idx, nodes):
        """
        Create a binary mask with True at index idx.
        
        Parameters:
        -----------
        idx : int
            The index to set to True.
        nodes : array-like
            The nodes array (used to determine mask size).
        
        Returns:
        --------
        mask : np.ndarray
            Boolean mask with True at idx position.
        """
        mask = np.zeros(len(nodes), dtype=bool)
        mask[idx] = True
        return mask

    @staticmethod
    def close_nodes_sample(graph: nx.Graph, k: int, seed: int = 42):
        """
        Select the highest degree node and k-1 randomly sampled neighbors from
        the graph. If the highest degree node has less than k-1 neighbors,
        include second-order neighbors until k nodes are selected.
        If there are multiple nodes with the same degree, the node with the 
        lowest index is selected.

        Parameters:
        -----------
        graph : nx.Graph
            The input graph.
        k : int
            The number of nodes to select.
        seed : int
            The random seed for reproducibility.

        Returns:
        --------
        sample : np.ndarray
            The position indices (0-based positions) of the selected nodes in the graph.
        """
        rng = np.random.default_rng(seed)
        # Create a mapping from node to its position in the graph
        node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
        
        degrees = dict(graph.degree())
        # Select highest degree node, breaking ties by lowest index
        max_degree = max(degrees.values())
        candidates = [node for node in degrees if degrees[node] == max_degree]
        max_degree_node = min(candidates, key=lambda node: node_to_index[node])
        neighbors = list(graph.neighbors(max_degree_node))
        
        if len(neighbors) >= k - 1:
            selected_neighbors = rng.choice(neighbors, size=k-1, replace=False)
            selected_nodes = [max_degree_node] + list(selected_neighbors)
        else:
            # If not enough neighbors, include second-order neighbors
            second_order_neighbors = set()
            for neighbor in neighbors:
                second_order_neighbors.update(graph.neighbors(neighbor))
            second_order_neighbors.discard(max_degree_node)  # Remove the max degree node itself
            all_candidates = list(set(neighbors) | second_order_neighbors)
            
            if len(all_candidates) >= k - 1:
                selected_candidates = rng.choice(all_candidates, size=k-1, replace=False)
                selected_nodes = [max_degree_node] + list(selected_candidates)
            else:
                raise ValueError("Not enough nodes to select from.")
        
        # Convert node identifiers to their position indices
        sample = np.array([node_to_index[node] for node in selected_nodes])
        return sample

    @staticmethod
    def farthest_nodes_sample(graph: nx.Graph, k: int, seed: int = 42,
                              distance_matrix: np.ndarray = None):
        """
        Select the k nodes that are farthest apart in the graph based on shortest path lengths.

        Parameters:
        -----------
        graph : nx.Graph
            The input graph.
        k : int
            The number of nodes to select.
        seed : int
            The random seed for reproducibility.
        distance_matrix : np.ndarray, optional
            Precomputed distance matrix for efficiency.

        Returns:
        --------
        sample : np.ndarray
            The indices of the selected nodes.
        """
        rng = np.random.default_rng(seed)
        if distance_matrix is None:
            distance_matrix = np.array(nx.floyd_warshall_numpy(graph))  
        else:
            distance_matrix = np.array(distance_matrix)
        
        if k > len(graph.nodes):
            raise ValueError("k cannot be greater than the number of nodes in the graph.")
        
        selected_nodes = []
        # Start with a random node
        first_node = rng.choice(len(graph.nodes))
        selected_nodes.append(first_node)
        
        while len(selected_nodes) < k:
            # Convert to array for proper indexing
            selected_array = np.array(selected_nodes)
            # Get maximum distance for each node from the selected nodes
            max_distances = np.max(distance_matrix[selected_array, :], axis=0)
            # Exclude already selected nodes
            max_distances[selected_nodes] = -np.inf 
            # Select the node with the maximum distance
            next_node = np.argmax(max_distances)
            selected_nodes.append(int(next_node))
        return np.array(selected_nodes)

    
    def sampling_energy(self,
                       n: int, 
                       k: int, 
                       gamma: float, 
                       mu: float, 
                       n_samples: int = 10000, 
                       seed: int = 42, 
                       n_workers: int = 8):
        """
        Sample the energy of a Hamiltonian for a given number of samples.

        Parameters:
        -----------
        n : int
            The number of elements in the system (in this case nodes).
        k : int
            The number of elements to choose in each sample.
        gamma : float
            A parameter for the Hamiltonian.
        mu : float
            A parameter for the Hamiltonian.
        n_samples : int
            The number of samples to generate.
        seed : int
            The random seed for reproducibility.
        n_workers : int
            The number of parallel workers to use. Uses ProcessPoolExecutor
            for true CPU parallelism (requires Hamiltonian to be picklable).

        Returns:
        --------
        hamiltonian : np.ndarray
            The sampled hamiltonian values.
        min_hamiltonian_sample : np.ndarray
            The sample with the minimum hamiltonian value.
        max_hamiltonian_sample : np.ndarray
            The sample with the maximum hamiltonian value.
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
                E = self.H.compute(sample, gamma=gamma, mu=mu)[0]
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
            # Use ProcessPoolExecutor instead of ThreadPoolExecutor for CPU-bound work
            # to avoid GIL contention. ThreadPoolExecutor would not provide real
            # parallelism since compute() is CPU-bound.
            try:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=len(tasks)) as executor:

                    futures = [
                        executor.submit(
                            worker, chunk_size, child_seed
                            ) for chunk_size, child_seed in tasks
                        ]
                    results = [future.result() for future in futures]
            except (TypeError, AttributeError, pickle.PicklingError) as e:
                # Fallback to sequential execution if Hamiltonian is not picklable
                print(f"Warning: ProcessPool failed ({type(e).__name__}). Falling back to sequential execution.")
                results = [worker(chunk_size, child_seed) for chunk_size, child_seed in tasks]

        energies = np.concatenate([r[0] for r in results])
        samples = np.concatenate([r[1] for r in results], axis=0)

        return energies, samples[energies.argmin()], samples[energies.argmax()]

    
    def _annealing_worker(self, S0_config, mu: float, gamma: float, Tmax: float, Tmin: float, 
                         cooling: float, chunk_size: int, seed: int, optimize: str = "minimize"):
        """
        Shared annealing worker function for both minimization and maximization.
        
        Parameters:
        -----------
        S0_config : np.ndarray
            Initial configuration.
        mu : float
            Parameter for the Hamiltonian.
        gamma : float
            Parameter for the Hamiltonian.
        Tmax : float
            Maximum temperature for annealing.
        Tmin : float
            Minimum temperature for annealing.
        cooling : float
            Cooling rate.
        chunk_size : int
            Number of iterations for this worker.
        seed : int
            Random seed for reproducibility.
        optimize : str
            Either "minimize" or "maximize".
        
        Returns:
        --------
        best_S : np.ndarray
            Best configuration found.
        best_E : float
            Best energy value.
        history : list
            Energy history.
        """
        rng = np.random.default_rng(seed)
        
        # Initial configuration
        S_current = S0_config.copy()
        E_current = abs(self.H.compute(S_current, mu=mu, gamma=gamma)[0])
        
        best_S = S_current.copy()
        best_E = E_current
        
        T = Tmax
        history = []
        
        # Annealing loop
        for _ in range(chunk_size):
            proposal_S = S_current.copy()
            occupied_indices = np.where(proposal_S == 1)[0]
            unoccupied_indices = np.where(proposal_S == 0)[0]
            
            # Swap move
            remove_node = rng.choice(occupied_indices)
            add_node = rng.choice(unoccupied_indices)
            
            proposal_S[remove_node] = 0
            proposal_S[add_node] = 1
            
            # Compute new energy
            proposal_E = abs(self.H.compute(proposal_S, mu=mu, gamma=gamma)[0])
            delta_E = proposal_E - E_current
            
            # Metropolis criterion
            if optimize == "minimize":
                if delta_E < 0:
                    accept = True
                else:
                    prob = np.exp(-delta_E / T)
                    accept = rng.random() < prob
                    
                # Update best solution
                if accept:
                    S_current = proposal_S
                    E_current = proposal_E
                    if E_current < best_E:
                        best_S = S_current.copy()
                        best_E = E_current
            else:  # maximize
                if delta_E > 0:
                    accept = True
                else:
                    prob = np.exp(delta_E / T)
                    accept = rng.random() < prob
                    
                # Update best solution
                if accept:
                    S_current = proposal_S
                    E_current = proposal_E
                    if E_current > best_E:
                        best_S = S_current.copy()
                        best_E = E_current
            
            history.append(E_current)
            
            # Cool down
            T *= cooling
            if T < Tmin:
                break
            elif proposal_E == 0.0:
                break
        
        return best_S, best_E, history
    
    def _run_parallel_annealing(self, S0_config, mu: float, gamma: float, Tmax: float, 
                               Tmin: float, cooloing: float, seed: int, steps: int, 
                               n_workers: int, optimize: str = "minimize"):
        """
        Execute parallel annealing runs.
        
        Parameters:
        -----------
        optimize : str
            Either "minimize" or "maximize".
        
        Returns:
        --------
        S_result : np.ndarray
            Best configuration.
        E_result : float
            Best energy value.
        history : list
            Energy history from best run.
        """
        n_workers = max(1, min(n_workers, steps))
        seed_seq = np.random.SeedSequence(seed)
        child_seeds = seed_seq.spawn(n_workers)
        chunk_sizes = [
            steps // n_workers + (1 if i < steps % n_workers else 0)
            for i in range(n_workers)
        ]
        
        tasks = [
            (chunk_sizes[i], child_seeds[i])
            for i in range(n_workers)
            if chunk_sizes[i] > 0
        ]
        
        def worker(chunk_size, child_seed):
            return self._annealing_worker(S0_config, mu, gamma, Tmax, Tmin, 
                                         cooloing, chunk_size, child_seed, optimize)
        
        if len(tasks) == 1:
            results = [worker(*tasks[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = [executor.submit(worker, chunk_size, child_seed) 
                          for chunk_size, child_seed in tasks]
                results = [future.result() for future in futures]
        # Select best result
        if optimize == "minimize":
            best_result = min(results, key=lambda x: x[1])
        else:  # maximize
            best_result = max(results, key=lambda x: x[1])
        return best_result[0], best_result[1], best_result[2]
    
    def min_energy_annealing(self, S0_config, mu: float = 1.0, gamma: float = 1.0,
                            Tmax: float = 1.0, Tmin: float = 1e-6, cooling: float = 0.995,
                            seed: int = 42, steps: int = 10000, n_workers: int = 8):
        """
        Minimize energy using simulated annealing with parallel execution.
        
        Parameters:
        -----------
        S0_config : np.ndarray
            Initial configuration (binary mask).
        mu : float
            Hamiltonian parameter.
        gamma : float
            Hamiltonian parameter.
        Tmax : float
            Maximum temperature.
        Tmin : float
            Minimum temperature.
        cooloing : float
            Cooling rate.
        seed : int
            Random seed.
        steps : int
            Number of annealing steps.
        n_workers : int
            Number of parallel workers.
        
        Returns:
        --------
        S_min : np.ndarray
            Configuration with minimum energy.
        E_min : float
            Minimum energy value.
        history : list
            Energy history.
        """
        return self._run_parallel_annealing(S0_config, mu, gamma, Tmax, Tmin, 
                                           cooling, seed, steps, n_workers, optimize="minimize")
    
    def max_energy_annealing(self, S0_config, mu: float = 1.0, gamma: float = 1.0,
                            Tmax: float = 1.0, Tmin: float = 1e-6, cooling: float = 0.995,
                            seed: int = 42, steps: int = 10000, n_workers: int = 8):
        """
        Maximize energy using simulated annealing with parallel execution.
        
        Parameters:
        -----------
        S0_config : np.ndarray
            Initial configuration (binary mask).
        mu : float
            Hamiltonian parameter.
        gamma : float
            Hamiltonian parameter.
        Tmax : float
            Maximum temperature.
        Tmin : float
            Minimum temperature.
        cooloing : float
            Cooling rate.
        seed : int
            Random seed.
        steps : int
            Number of annealing steps.
        n_workers : int
            Number of parallel workers.
        
        Returns:
        --------
        S_max : np.ndarray
            Configuration with maximum energy.
        E_max : float
            Maximum energy value.
        history : list
            Energy history.
        """
        return self._run_parallel_annealing(S0_config, mu, gamma, Tmax, Tmin, 
                                           cooling, seed, steps, n_workers, optimize="maximize")
if __name__ == "__main__":
    
    graph = nx.erdos_renyi_graph(n=300, p=0.1, seed=42)

    H = Hamiltonian(graph)
    optimizer = EnergyOptimizer(H)
    distance_matrix = nx.floyd_warshall_numpy(graph)

    n = 100
    k = 10
    gamma = 0.5
    mu = 0.1
    n_samples = 100000
    seed = 42
    n_workers = os.cpu_count() or 1

    hamiltonian, min_hamiltonian_sample, max_hamiltonian_sample = optimizer.sampling_energy(
        n, k, gamma, mu, n_samples, seed, n_workers
    )

    farthest_sample = optimizer.farthest_nodes_sample(graph, k, seed, distance_matrix=distance_matrix)
    close_sample = optimizer.close_nodes_sample(graph, k, seed)
    
    # Convert sample indices to binary mask for annealing
    S0_mask = np.zeros(n, dtype=int)
    S0_mask[farthest_sample] = 1

    maxE = optimizer.max_energy_annealing(
    S0_mask, gamma=gamma, mu=mu, steps=10000, n_workers=8, seed=42,
    Tmin=1e-6)
    print(f"Minimum hamiltonian sample: {min_hamiltonian_sample}, Hamiltonian: {hamiltonian.min()}")
    print(f"Maximum hamiltonian sample: {max_hamiltonian_sample}, Hamiltonian: {hamiltonian.max()}")
    print(f"Farthest nodes sample: {farthest_sample}")
    print(f"Close nodes sample: {close_sample}")
    print(f"Max energy annealing result: {maxE[1]}")