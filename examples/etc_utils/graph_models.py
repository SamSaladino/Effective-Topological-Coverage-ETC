"""Graph generation and visualization helpers used by the examples notebooks.           

Provides:
- G_string_of_pearls: build the 'string of pearls' support to create the Barbell graph.
- connect_components: lightly connect components so a graph is connected.
- smart_layout: pick a layout depending on graph size.
- show_graph: quick visualization for a single graph.
- show_highlighted_subsets: visualize one or more subsets with customizable legend.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def G_string_of_pearls(m: int = 7, clique: int = 8) -> nx.Graph:
    """Create a "string of pearls" graph: m cliques of size `clique` 
    connected in a path.
    ------------
    Parameters:
    - m: number of cliques
    - clique: size of each clique
    -----------
    Returns:
     - networkx.Graph: integer node labels (0..n-1) ordered sorted.
    """
    parts = [nx.complete_graph(clique) for _ in range(m)]
    G = nx.disjoint_union_all(parts)
    offset = 0
    for i in range(m - 1):
        a = offset + clique - 1
        b = offset + clique
        G.add_edge(a, b)
        offset += clique
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def connect_components(G: nx.Graph, seed: int = 1) -> nx.Graph:
    """If G is disconnected, add edges between components so the result is connected.

    The operation mutates a copy of the input graph and returns it with integer
    node labels (0..n-1) sorted. Uses the provided seed for deterministic choices.
    """
    Gc = G.copy()
    if nx.is_connected(Gc):
        return nx.convert_node_labels_to_integers(Gc, ordering="sorted")
    rng = random.Random(seed)
    components = [list(c) for c in nx.connected_components(Gc)]
    for i in range(len(components) - 1):
        u = rng.choice(components[i])
        v = rng.choice(components[i + 1])
        Gc.add_edge(u, v)
    return nx.convert_node_labels_to_integers(Gc, ordering="sorted")


def smart_layout(G: nx.Graph, seed: int = 1):
    """Return a layout for plotting depending on graph size.

    - For n <= 1200: try Kamada-Kawai, fall back to spring_layout with seed.
    - For larger graphs: spectral_layout.
    """
    n = G.number_of_nodes()
    if n <= 500:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=seed)
    return nx.spectral_layout(G)


def show_graph(G: nx.Graph, title: Optional[str] = None, node_size: int = 8, 
               ax: Optional[plt.Axes] = None):
    """Simple graph visualization using a smart layout.

    If `ax` is provided the drawing is performed on it; otherwise a new figure
    is created and shown.
    """
    pos = smart_layout(G)
    created_fig = False
    if ax is None:
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        created_fig = True
    nx.draw_networkx_nodes(G, pos, node_size=node_size, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.3, ax=ax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    if created_fig:
        plt.show()


def show_highlighted_subsets(
    G: nx.Graph,
    subsets: Sequence[Iterable],
    titles: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    legend_labels: Optional[Sequence[str]] = None,
    node_size_bg: int = 50,
    node_size_highlight: int = 110,
    edge_alpha: float = 0.65,
    edge_width: float = 0.6,
    figsize: Tuple[int, int] = (12, 6),
    pos: Optional[dict] = None,
):
    """Visualize one or more highlighted node subsets side-by-side.

    Parameters
    - G: graph to visualize.
    - subsets: sequence of iterables of node ids (int or str) to highlight. 
    One subplot per subset.
    - titles: optional titles per subplot. 
    If missing, simple "Subset i" labels are used.
    - colors: optional colors per subplot. 
    Defaults to a colorblind-friendly pair repeated.
    - legend_labels: optional labels for legend; 
    must match the meaning of the highlighted nodes.
    - pos: optional layout dictionary (node->(x,y)). 
    If None, smart_layout is computed once.

    Returns (fig, axes).
    """
    k = len(subsets)
    if k == 0:
        raise ValueError("`subsets` must contain at least one subset to highlight")

    if titles is None:
        titles = [f"Subset {i}" for i in range(k)]
    if colors is None:
        base_colors = ["#D55E00", "#0072B2", "#009E73", "#F0E442"]
        # repeat colors if fewer provided than subsets
        colors = [base_colors[i % len(base_colors)] for i in range(k)]

    if pos is None:
        pos = smart_layout(G)

    fig, axes = plt.subplots(1, k, figsize=figsize, constrained_layout=True)
    if k == 1:
        axes = [axes]

    edge_color = "#888888"

    for ax, subset, title, color in zip(axes, subsets, titles, colors):
        ax.set_title(title)
        ax.axis("off")
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_width, alpha=edge_alpha, edge_color=edge_color)

        subset_set = set(subset)
        background = [n for n in G.nodes() if n not in subset_set]
        nx.draw_networkx_nodes(
            G, pos, nodelist=background, 
            node_color="#DDDDDD", node_size=node_size_bg, linewidths=0, ax=ax)

        highlight_nodes = [n for n in G.nodes() if n in subset_set]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=highlight_nodes,
            node_color=color,
            node_size=node_size_highlight,
            edgecolors="black",
            linewidths=0.6,
            ax=ax,
        )

    # optional legend on first axis
    if legend_labels:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", 
                   markerfacecolor="#DACDCD", markersize=6, 
                   label="Background", markeredgecolor="#DDDDDD"),
        ]
        # add one representative for highlighted (use first color)
        handles.append(Line2D([0], [0], marker="o", color="w", 
                              markerfacecolor=colors[0], markeredgecolor="k", 
                              markersize=8, label=legend_labels[0])
                              )
        axes[0].legend(handles=handles, loc="lower left", frameon=False)

    plt.show()
    return fig, axes


def get_graphs_list(seed: int = 2):
    """Return a list of (name, graph) tuples used in examples notebooks.

    The graphs are produced in the same order as the notebook and will be
    connected using `connect_components` where appropriate.
    """
    graphs = [
        ("Barbell(c=25, p=5)", nx.barbell_graph(25, 5)),
        ("Lollipop(c=30, t=50)", nx.lollipop_graph(30, 50)),
        ("String of Pearls(m=7,c=8)", G_string_of_pearls(7, 8)),
        ("Path(n=100)", nx.path_graph(100)),
        ("Balanced Tree(r=3,h=5)", nx.balanced_tree(3, 5)),
        ("Core-Periphery", connect_components(
            nx.stochastic_block_model([80, 220], [[0.18, 0.05],[0.05, 0.01]], 
                                                  seed=seed), 
                                                  seed=seed)
            ),
        ("Erdos-Renyi(n=200,p=0.05)", connect_components(
            nx.erdos_renyi_graph(200, 0.05, seed=seed), seed=seed)),
        ("Random Partition (communities)", connect_components(
            nx.random_partition_graph([15, 25, 20], 0.8, 0.02, seed=seed), seed=seed)),
        ("2D Grid 20x20", nx.grid_2d_graph(20, 20)),
        ("Toroidal Grid 20x20", nx.grid_2d_graph(20, 20, periodic=True)),
        ("Circular Ladder(n=100)", nx.circular_ladder_graph(50)),
        ("Circulant(n=200,steps=[1,2,3])", nx.circulant_graph(200, [1, 2, 3])),
        ("WS beta=0.0", connect_components(
            nx.watts_strogatz_graph(150, 8, 0.0, seed=seed), seed=seed)),
        ("WS beta=0.2", connect_components(
            nx.watts_strogatz_graph(150, 8, 0.2, seed=seed), seed=seed)),
        ("WS beta=1.0", connect_components(
            nx.watts_strogatz_graph(150, 8, 1.0, seed=seed), seed=seed)),
        ("BA(n=400,m=2)", nx.barabasi_albert_graph(100, 2, seed=seed)),
        ("Powerlaw-Cluster", connect_components(
            nx.powerlaw_cluster_graph(100, 2, 0.3, seed=seed), seed=seed)),
        ("d-Regular(n=400,d=6)", connect_components(
            nx.random_regular_graph(6, 100, seed=seed), seed=seed)),
        ("Geometric(n=400,r=0.07)", connect_components(
            nx.random_geometric_graph(100, 0.07, seed=seed), seed=seed)),
        ("Star(n=400)", nx.star_graph(50)),
    ]
    return graphs
