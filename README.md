Effective Topological Coverage or Etcetera ;p (ETC) is a Python framework for evaluating how a set of observed metabolites is distributed across a metabolic network.

The method distinguishes between metabolite sets that are concentrated in a small biochemical region and sets that are broadly dispersed across the network. It combines local biochemical connectivity and global network dispersion in a single energy-based formulation.

ETC was developed for the comparison of metabolite lists obtained from different LC-MS analytical workflows, but the framework can be applied to any set of observations mapped to a graph.

## Scientific publication

Here will be when published ;p

## Method

The metabolic network is represented as an undirected, unweighted compound graph

$$
G=(V,E),
$$

where:

* (V) is the set of metabolites;
* (E) contains an edge between two metabolites when they participate as substrate and product in the same reaction.

For a network containing (n) metabolites, an observed metabolite set is represented by a binary vector

$$
\mathbf{s}\in{0,1}^{n},
$$

where (s_i=1) when metabolite (i) is observed and (s_i=0) otherwise.

The H function is defined as

$$
\mathcal{H}(\mathbf{s})
=======================

-\mu
\sum_{i<j}
A_{ij}s_i s_j
+
\gamma
\sum_{i<j}
\frac{(1-A_{ij})s_i s_j}{d_{ij}^{2}},
$$

where:

* (A_{ij}) is the adjacency matrix;
* (d_{ij}) is the shortest-path distance between nodes (i) and (j);
* (\mu>0) controls the contribution of local biochemical connections;
* (\gamma>0) controls the contribution of global dispersion.

The first term is

$$
T_1
===

-\mu
\sum_{i<j}
A_{ij}s_i s_j.
$$

It becomes more negative when many observed metabolites are directly connected.

The second term is

$$
T_2
===

\gamma
\sum_{i<j}
\frac{(1-A_{ij})s_i s_j}{d_{ij}^{2}}.
$$

It evaluates non-adjacent observed pairs. Directly connected pairs are excluded through the factor (1-A_{ij}).

The complete Hamiltonian is

$$
\mathcal{H}=T_1+T_2.
$$

Its sign provides information about the dominant topological regime:

* (\mathcal{H}<0): local connectivity dominates and the observations tend to cluster;
* (\mathcal{H}>0): non-local dispersion dominates;
* (\mathcal{H}\approx0): local coherence and global dispersion are approximately balanced.

The effective energy is

$$
E=|\mathcal{H}|.
$$

A small value of (E) identifies a configuration close to the balance between clustering and dispersion.

## Suggested overview figure

Workflow figure should be here.

```markdown
![Overview of the ETC workflow](docs/figures/etc_workflow.png)
```


> Overview of the Effective Topological Coverage workflow. Metabolite annotations obtained from different analytical workflows are mapped to a common metabolic compound graph. Local adjacency and global shortest-path distances are combined in the Hamiltonian (\mathcal{H}). The effective energy (E=|\mathcal{H}|) quantifies the balance between local biochemical coherence and global network dispersion.

Avoid using a generic network screenshot. The figure should explain the method, not merely decorate the README.

## Repository structure

```text
Effective-Topological-Coverage-ETC/
├── src/
│   └── etc/
│       ├── __init__.py
│       ├── cli.py
│       ├── hamiltonian.py
│       ├── optimization.py
│       ├── landscape_diagrams.py
│       └── visualization.py
├── scripts/
│   ├── CompGraph_met4j.sh
│   └── extract_inchikeys_from_metanx.py
├── notebooks/
│   ├── 00_small_graphs_figures.ipynb
│   ├── 01_Daniel_compilation.ipynb
│   └── 02_Robustness_study.ipynb
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

The core implementation is located in `src/etc/`. The notebooks reproduce the analyses and figures, while the scripts contain data-preparation utilities that are not part of the core Python API.

## Requirements

The core package requires:

* Python 3.12 or later;
* NumPy;
* SciPy;
* NetworkX;
* pandas;
* Matplotlib.

The exact landscape solver additionally requires OR-Tools.

Java and Met4J are only required when rebuilding a compound graph from an SBML metabolic reconstruction.

## Installation

Clone the repository:

```bash
git clone https://github.com/SamSaladino/Effective-Topological-Coverage-ETC.git
cd Effective-Topological-Coverage-ETC
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the core package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation means that changes made inside `src/etc/` are immediately available without reinstalling the package.

Install development and testing dependencies with:

```bash
python -m pip install -e ".[dev]"
```

Install the optional exact landscape solver with:

```bash
python -m pip install -e ".[landscape]"
```

Verify the installation:

```bash
python -c "import etc; print(etc.__file__)"
```

The command-line interface can be checked with:

```bash
etc --help
```

## Input requirements

### Graph

ETC expects a simple, undirected graph.

The command-line interface currently accepts:

* GraphML files;
* GML files.

The graph should satisfy the following conditions:

* each node represents one metabolite;
* edges represent direct biochemical relationships;
* parallel edges have already been merged;
* isolated or unwanted nodes have been handled during preprocessing;
* node identifiers are unique;
* the graph is undirected;
* the graph is not a multigraph.

The interpretation of the results depends on graph preprocessing. Side-compound removal, compartment merging, component selection, and degree-based filtering should therefore be documented for every analysis.

### Observations

The command-line interface expects a text file containing one observed node identifier per line.

Example:

```text
M_glucose_c
M_lactate_c
M_pyruvate_c
```

Empty lines and lines beginning with `#` are ignored.

Every identifier in the observation file must match a node identifier in the graph exactly.

Duplicate identifiers should not be included.

## Command-line usage

The `evaluate` command calculates (\mathcal{H}), (T_1), (T_2), and (E) for one observed metabolite set.

```bash
etc evaluate \
    --graph data/network.graphml \
    --observations data/observed_metabolites.txt \
    --mu 0.9 \
    --gamma 0.01
```

Example output:

```json
{
  "n": 3200,
  "k": 75,
  "H": -2.3417,
  "T1": -5.1200,
  "T2": 2.7783,
  "E": 2.3417
}
```

The output fields are:

* `n`: number of nodes in the graph;
* `k`: number of observed metabolites;
* `H`: signed Hamiltonian;
* `T1`: local adjacency contribution;
* `T2`: global non-adjacent contribution;
* `E`: effective energy, calculated as `abs(H)`.

The sign of `H` should not be discarded when interpreting the topology. Two configurations may have the same effective energy but belong to opposite topological regimes.

## Python usage

### Evaluate an observed metabolite set

```python
import networkx as nx
import numpy as np

from etc import Hamiltonian
```

Load the graph:

```python
graph = nx.read_graphml("data/network.graphml")
```

Create the Hamiltonian object:

```python
hamiltonian = Hamiltonian(graph)
```

Create a mapping between graph node identifiers and their positional indices:

```python
node_to_index = {
    node: index
    for index, node in enumerate(graph.nodes())
}
```

Convert observed node identifiers to positional indices:

```python
observed_nodes = [
    "M_glucose_c",
    "M_lactate_c",
    "M_pyruvate_c",
]

observed_indices = np.asarray(
    [node_to_index[node] for node in observed_nodes],
    dtype=int,
)
```

Evaluate the configuration:

```python
H_value, T1, T2 = hamiltonian.compute(
    observed_indices,
    mu=0.9,
    gamma=0.01,
)

energy = abs(H_value)

print(f"H = {H_value:.6f}")
print(f"T1 = {T1:.6f}")
print(f"T2 = {T2:.6f}")
print(f"E = {energy:.6f}")
```

Important: `Hamiltonian.compute()` expects positional node indices. It does not expect graph node labels or a binary mask.

### Minimal reproducible example

```python
import networkx as nx

from etc import Hamiltonian

graph = nx.path_graph(3)
hamiltonian = Hamiltonian(graph)

H_value, T1, T2 = hamiltonian.compute(
    [0, 2],
    mu=2.0,
    gamma=4.0,
)

print(H_value, T1, T2)
```

For the path

```text
0 -- 1 -- 2
```

nodes 0 and 2 are separated by distance 2 and are not adjacent. Therefore,

$$
T_1=0,
$$

and

$$
T_2=\frac{4}{2^2}=1.
$$

The expected result is:

```text
1.0 0.0 1.0
```

## Sampling the signed Hamiltonian

`EnergyOptimizer.sampling_h()` generates random subsets containing exactly (k) nodes and returns their signed Hamiltonian values.

```python
from etc import EnergyOptimizer

optimizer = EnergyOptimizer(hamiltonian)

h_values, min_h_sample, max_h_sample = optimizer.sampling_h(
    n=hamiltonian.n,
    k=20,
    gamma=0.01,
    mu=0.9,
    n_samples=10_000,
    seed=42,
    n_workers=8,
)
```

The returned values are:

* `h_values`: signed Hamiltonian values for all sampled subsets;
* `min_h_sample`: node indices of the most negative sampled configuration;
* `max_h_sample`: node indices of the most positive sampled configuration.

This function samples signed (\mathcal{H}), not (E=|\mathcal{H}|).

Therefore:

```python
h_values.argmin()
```

identifies the most attraction-dominated sampled configuration, whereas:

```python
np.abs(h_values).argmin()
```

identifies the sampled configuration closest to the balance point (\mathcal{H}=0).

## Simulated annealing

The annealing methods operate on binary masks.

A binary mask must:

* have length equal to the number of graph nodes;
* contain only zeros and ones;
* contain exactly (k) selected positions.

Create an initial mask:

```python
import numpy as np

initial_indices = np.array([0, 4, 7, 12])

initial_mask = np.zeros(
    hamiltonian.n,
    dtype=int,
)

initial_mask[initial_indices] = 1
```

### Minimize the effective energy

```python
best_mask, E_min, history = optimizer.min_energy_annealing(
    S0_config=initial_mask,
    mu=0.9,
    gamma=0.01,
    steps=10_000,
    n_workers=8,
    seed=42,
)
```

Convert the returned mask back to positional indices:

```python
best_indices = np.flatnonzero(best_mask)
```

Validate the result:

```python
H_min, T1_min, T2_min = hamiltonian.compute(
    best_indices,
    mu=0.9,
    gamma=0.01,
)

assert np.isclose(E_min, abs(H_min))
assert best_mask.sum() == initial_mask.sum()
```

### Maximize the effective energy

```python
worst_mask, E_max, history_max = optimizer.max_energy_annealing(
    S0_config=initial_mask,
    mu=0.9,
    gamma=0.01,
    steps=10_000,
    n_workers=8,
    seed=42,
)
```

The minimum and maximum annealing methods optimize

$$
E=|\mathcal{H}|,
$$

not the signed Hamiltonian.

The selected-node count (k) is preserved through one-for-one swap moves.

## Normalized coverage

After estimating (E_{\min}) and (E_{\max}), normalized coverage can be calculated as

$$
C
=

1-
\frac{E-E_{\min}}
{E_{\max}-E_{\min}}.
$$

Example:

```python
E_observed = abs(H_value)

if np.isclose(E_max, E_min):
    raise ValueError(
        "Coverage cannot be normalized because "
        "E_max and E_min are equal."
    )

coverage = 1.0 - (
    (E_observed - E_min)
    / (E_max - E_min)
)

print(f"Coverage = {coverage:.6f}")
```

Do not automatically clip the result to ([0,1]).

A result outside this interval indicates that the estimated extrema do not bound the observed configuration. This may occur when stochastic optimization has not adequately explored the configuration space.

Values of (E_{\min}), (E_{\max}), and (C) are only comparable when they were obtained using the same:

* graph;
* preprocessing procedure;
* number of selected nodes (k);
* values of (\mu) and (\gamma);
* optimization definition.

## Exact landscape analysis

The optional landscape module provides exact or approximate methods for exploring small configuration spaces.

Install OR-Tools:

```bash
python -m pip install -e ".[landscape]"
```

Example imports:

```python
from etc.landscape_diagrams import (
    build_Jij,
    sample_k_closest_to_zero,
    solve_extreme_k,
)
```

Exact optimization is intended primarily for toy graphs and small validation problems. It may become computationally expensive for large metabolic networks.

For Human1-scale analyses, stochastic sampling and simulated annealing are generally more practical.

## Building the compound graph with Met4J

The script

```text
scripts/CompGraph_met4j.sh
```

can be used to generate a compound graph from an SBML metabolic reconstruction.

Example:

```bash
scripts/CompGraph_met4j.sh \
    --jar /path/to/met4j-toolbox.jar \
    --sbml /path/to/Human-GEM.xml \
    --output-dir data/generated
```

The script:

1. identifies side compounds;
2. generates a compound graph without side compounds;
3. optionally merges metabolite compartments by name;
4. writes the resulting graph files to the selected output directory.

This step requires Java and a compiled Met4J toolbox JAR.

The exact Human-GEM release, Met4J version, side-compound procedure, compartment-merging rule, and graph filtering steps should be recorded in the analysis notebook or accompanying metadata.

## Reproducibility notebooks

### `00_small_graphs_figures.ipynb`

Introduces the behavior of the Hamiltonian on small graph topologies.

This notebook should be used to verify the interpretation of:

* clustered configurations;
* dispersed configurations;
* balanced configurations;
* the effects of (k), (\mu), and (\gamma).

### `01_Daniel_compilation.ipynb`

Contains the main Human1 analysis and comparisons between analytical workflows or laboratories.

The notebook includes:

* loading the processed Human1 graph;
* mapping metabolite annotations;
* evaluating signed (\mathcal{H});
* random sampling;
* annealing;
* estimation of energy bounds;
* comparison between observed metabolite sets.

### `02_Robustness_study.ipynb`

Evaluates whether the conclusions are stable under changes in model parameters, optimization seeds, and observed metabolite composition.

The robustness analysis should include:

1. repeated optimization across random seeds;
2. sensitivity to (\mu) and (\gamma);
3. sensitivity to the number of annealing steps;
4. node-replacement perturbations that preserve (k);
5. comparison with random subsets of equal size;
6. sensitivity to graph preprocessing.

A robustness result should report both the variation in signed (\mathcal{H}) and the variation in (E=|\mathcal{H}|).

## Testing

Run the complete test suite with:

```bash
python -m pytest
```

Run only the Hamiltonian tests with:

```bash
python -m pytest tests/test_hamiltonian_class.py -v
```

Run only optimization tests with:

```bash
python -m pytest tests/test_optimization.py -v
```

Run command-line interface tests with:

```bash
python -m pytest tests/test_cli.py -v
```

The continuous-integration workflow installs the package in a clean Python environment and runs the same tests automatically for pushes and pull requests.

## Interpretation and limitations

ETC is a topological metric. It does not directly account for:

* metabolite concentrations;
* measurement uncertainty;
* reaction fluxes;
* reaction directionality;
* stoichiometric coefficients;
* tissue-specific activity;
* metabolite compartment abundance;
* annotation-confidence differences.

The compound graph is undirected and unweighted. This representation is appropriate for evaluating binary topological coverage, but it does not preserve all biochemical information from the genome-scale metabolic model.

The method is also sensitive to:

* side-compound removal;
* graph connectivity;
* node-identifier mapping;
* distance definition;
* the selected values of (\mu) and (\gamma);
* the number of observations (k);
* the quality of stochastic extrema estimation.

Results should therefore be interpreted as network-topological coverage under a specified graph construction and parameterization, not as a complete measure of biochemical observability.

## Citation

A software citation and manuscript citation should be added after publication.

Suggested temporary format:

```text
Costa, S. Effective Topological Coverage: an energy-based framework
for evaluating metabolite distributions in metabolic networks.
Version X.Y.Z.
```

Add the final DOI, journal reference, and software archive link when available.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

The software was developed by Sandra Costa during her PhD research at INRAE Toulouse within the HUMAN Marie Skłodowska-Curie Doctoral Network.

The final copyright-holder wording should be confirmed with the relevant institutional contact before a formal public release.

## Acknowledgments

This work was developed within the HUMAN Doctoral Network and supported by the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Actions as part of my PhD project.

Supervisors: Fabien Jourdan and Clément Frainay (INRAE), MeTExplore, MeT4J, Collaborator institutions that produced and collected the datastes: HMGU, CEMBIO, AFEKTA, ICL, AUTh.
