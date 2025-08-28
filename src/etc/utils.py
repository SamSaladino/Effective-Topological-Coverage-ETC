from typing import Optional
import numpy as np
import networkx as nx
import random

class GraphLabelPropagation:
    """Lightweight utilities for label-propagation experiments on NetworkX graphs.

    This class provides a simple custom LPA and helpers to set random labels.
    Visualization and scikit-learn integration were removed to keep the module
    lightweight for unit testing and linting.
    """

    def __init__(self, graph: nx.Graph):



        class GraphLabelPropagation:
            """Lightweight utilities for label-propagation experiments 
            on NetworkX graphs.

            This class provides a simple custom LPA and helpers to set random labels.
            """

            def __init__(self, graph: nx.Graph):
                self.graph = graph
                self.labels = None

            def set_random_labels(
                    self, label_prob: float = 0.2, seed: Optional[int] = None
                    ):
                """Randomly assign binary labels to nodes.

                Returns a dict mapping node -> {0,1}.
                """
        from typing import Optional

        import numpy as np
        import networkx as nx
        import random


        class GraphLabelPropagation:
            """Lightweight utilities for label-propagation experiments on NetworkX graphs.

            This class provides a simple custom LPA and helpers to set random labels.
            """

            def __init__(self, graph: nx.Graph):
                self.graph = graph
                self.labels = None

            def set_random_labels(self, label_prob: float = 0.2, seed: Optional[int] = None):
                """Randomly assign binary labels to nodes.

                Returns a dict mapping node -> {0,1}.
                """
                if seed is not None:
                    random.seed(seed)
                self.labels = {
                    node: 1 if random.random() < label_prob else 0
                    for node in self.graph.nodes()
                }
                return self.labels

            def custom_lpa(self, iterations: int = 1000):
                """Simple label propagation using NumPy matrix operations.

                This is deliberately minimal and aimed at being deterministic for tests.
                """
                labels = np.array([self.labels[node] for node in self.graph.nodes()])
                adj_matrix = nx.to_numpy_array(self.graph)

                for _ in range(iterations):
                    label_counts = adj_matrix @ labels
                    labels = np.where(label_counts > 0, 1, 0)

                self.labels = {node: int(labels[i]) for i, node in enumerate(self.graph.nodes())}
                return self.labels
