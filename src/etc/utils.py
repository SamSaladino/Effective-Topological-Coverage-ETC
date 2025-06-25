import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.semi_supervised import LabelPropagation

class GraphLabelPropagation:
    def __init__(self, graph):
        """
        Initialize with a NetworkX graph.
        """
        self.graph = graph
        self.labels = None

    def set_random_labels(self, label_prob=0.2, seed=None):
        """
        Randomly assign labels: 1 with probability label_prob, else 0.
        """
        if seed is not None:
            random.seed(seed)
        self.labels = {node: 1 if random.random() < label_prob else 0 for node in self.graph.nodes()}
        return self.labels

    def plot(self, labels=None, title="Graph Label Visualization"):
        """
        Plot the graph with node colors based on labels.
        """
        if labels is None:
            labels = self.labels
        color_map = {0: "gray", 1: "blue"}
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 8))
        colors = [color_map[labels[node]] for node in self.graph.nodes()]
        nx.draw(self.graph, pos, node_color=colors, with_labels=True, font_weight='bold', cmap=plt.cm.viridis)
        nx.draw_networkx_labels(self.graph, pos, {node: labels[node] for node in self.graph.nodes() if labels[node] == 1})
        plt.title(title)
        plt.show()

    def custom_lpa(self, iterations=1000):
        """
        Custom Label Propagation Algorithm.
        """
        labels = self.labels.copy()
        for _ in range(iterations):
            for node in self.graph.nodes():
                label_counts = {}
                for neighbor in self.graph.neighbors(node):
                    label = labels[neighbor]
                    label_counts[label] = label_counts.get(label, 0) + 1
                max_count = 0
                max_labels = []
                for label, count in label_counts.items():
                    if count > max_count:
                        max_count = count
                        max_labels = [label]
                    elif count == max_count:
                        max_labels.append(label)
                if max_labels:
                    labels[node] = int(max_labels[0])
        self.labels = labels
        return labels

    def sklearn_lpa(self, max_iter=500):
        """
        Use scikit-learn's LabelPropagation.
        """
        adj_matrix = nx.to_numpy_array(self.graph)
        initial_labels = np.array([1 if self.labels[node] == 1 else -1 for node in self.graph.nodes()])
        model = LabelPropagation(max_iter=max_iter, kernel='rbf')
        model.fit(adj_matrix, initial_labels)
        final_labels = model.transduction_
        labels_dict = {node: int(final_labels[node]) for node in self.graph.nodes()}
        return labels_dict

# Example usage:
if __name__ == "__main__":
    # Create the Karate Club graph
    G = nx.karate_club_graph()
    glp = GraphLabelPropagation(G)

    # Set random initial labels
    glp.set_random_labels(label_prob=0.2, seed=42)
    glp.plot(title="Initial Random Labels")

    # Run custom LPA
    lpa_labels = glp.custom_lpa(iterations=100)
    glp.plot(labels=lpa_labels, title="Labels After Custom LPA")

    # Run sklearn LPA
    sklearn_labels = glp.sklearn_lpa(max_iter=500)
    glp.plot(labels=sklearn_labels, title="Labels After sklearn LabelPropagation")