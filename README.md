# Incremental Graph Neural Networks for Dynamic Channel Assignment

**Sole-authored research by Pavan Parola · IEEE SPERT 2025**  
[IEEE Xplore (DOI: 10.1109/SPERT67079.2025.11469727)](https://doi.org/10.1109/SPERT67079.2025.11469727) · [Implementation](dynamic_gnn.py) · [Notebook](Incremental_GNN_Combinatorial_Optimization.ipynb)

An experimental framework for **dynamic graph coloring**: maintain a valid assignment of radio channels as an interference graph changes, balancing interference avoidance, reassignment overhead, computation time, and spectrum usage.

> **Research integrity note.** The published paper and this evolving repository should not be treated as identical experimental artifacts. The current code includes subsequent corrections to conflict accounting, repair, held-out evaluation, baseline implementations, and benchmark methodology. Published figures are **not claimed to be reproduced** by the current repository without rerunning the corresponding version and documenting the environment. All networks generated here are synthetic; random-geometric graphs are spatial proxies, **not measured HAN/NAN deployments**.

## Problem

Represent an interference network as an undirected graph \(G_t = (V,E_t)\). Adjacent nodes cannot occupy the same channel. When edges are added or removed, a coloring must adapt while minimizing:

- **Interference:** adjacent vertices assigned identical channels (hard validity requirement in final output).
- **Switching cost:** number of nodes whose channel changes compared with the previous time step.
- **Spectrum demand:** distinct channels used.
- **Update latency:** wall-clock computation time per update batch.

This is an evolving, combinatorial optimization problem. It illustrates the trade-off between **global optimization** (which may disrupt many assignments) and **local incremental adjustment** (which favors stability).

## Approach

`dynamic_gnn.py` contains:

1. **Graph representation and structural features:** normalized degree, local clustering, two-hop neighborhood coverage, and scaled degree, concatenated with the previous assignment's one-hot channel encoding.
2. **Model:** two GraphSAGE layers followed by a two-head graph-attention layer; a skip-connected classification head predicts per-node channel logits.
3. **Training:** supervised initialization against DSATUR graph colorings, followed by incremental training on random edge-update sequences. Cross-entropy is combined with a differentiable neighbor-conflict penalty; nodes near changes receive additional emphasis.
4. **Hard-validity post-processing:** predicted assignments are greedily repaired and checked against **all unique undirected edges**; invalid output raises an exception.
5. **Baselines:** DSATUR, Welsh–Powell/largest-first, incremental greedy repair, and simulated annealing with cooling and Metropolis acceptance. Global recolorings have their arbitrary color labels aligned to the prior assignment to avoid counting pure renamings as switches.

The final conflict-free assignment is the result of **neural predictions plus deterministic repair**. Zero conflicts alone do not establish that the neural network learned a valid coloring without repair.

## Reproduce the current implementation

Use Python **3.11**. A CPU environment is sufficient for smoke tests; a CUDA GPU is useful for the full two-phase training run.

```bash
git clone https://github.com/YaolaP4/Incremental_GNN_For_Combinatorial_Problems.git
cd Incremental_GNN_For_Combinatorial_Problems
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a CUDA installation, follow the matching [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your system, then install the remaining dependencies. These are **compatibility ranges**, not a validated lockfile; record exact resolved package versions, GPU, CUDA driver, and hardware before comparing timings across machines.

Open [`Incremental_GNN_Combinatorial_Optimization.ipynb`](Incremental_GNN_Combinatorial_Optimization.ipynb) and run cells from top to bottom. It performs:

- Deterministic seeding (`set_seed(42)`) and a conflict-count/repair sanity check.
- Training-set generation (20 Erdős–Rényi, 15 Barabási–Albert, 15 Watts–Strogatz, plus Petersen and grid controls).
- Two-phase training: **20 static epochs** and **30 incremental epochs**.
- Evaluation on separately generated medium-sized graphs, 500-node random-geometric graphs, and 1,000-node graphs, with **10 update rounds per graph**.
- Save of learned model weights to `incremental_gnn_corrected.pt`.

The generated training/evaluation inputs are reproducible from seeds, but exact floating-point training and timings may vary by device, versions, and available hardware. The notebook's installation cell currently installs packages without an exact lockfile; for environment-controlled runs, use the commands above and run the notebook using that environment.

### Small code-level check

```python
import networkx as nx
import torch
from dynamic_gnn import graph_to_edge_index, count_conflicts, post_process_coloring

graph = nx.path_graph(5)
edge_index = graph_to_edge_index(graph, torch.device("cpu"))
pred = torch.tensor([0, 1, 1, 0, 0])
assert count_conflicts(edge_index, pred) == 2
assert count_conflicts(edge_index, post_process_coloring(pred, edge_index, 5)) == 0
```

## Evaluation protocol

The code benchmarks **five algorithms on the same sequence of graph updates**. Each method retains its own prior coloring. In each round, the update sampler attempts 3–5 changes, with a 70% probability of attempting edge addition and 30% removal; unsuccessful additions/removals may lead to fewer actual changes. Conflicts are verified over unique undirected edges. The reported columns are:

| Metric | Meaning |
| --- | --- |
| `time_ms` | Measured processing time per update batch, in milliseconds |
| `conflicts` | Number of final edges whose endpoints share a channel; each undirected edge counted once |
| `changes` | Number of nodes whose final channel differs from that method's own preceding step |
| `colors` | Number of distinct channels in the final assignment |

In the updated benchmark, the GNN timing includes incremental feature preparation, forward inference, prediction repair, and GPU synchronization (when applicable). The *other* baselines' timings cover their coloring routine; graph update construction, shared orchestration, and final result aggregation are outside every individual method's timing. **Reported timing should therefore be interpreted as method-specific compute, not end-to-end system latency**, and cross-hardware comparisons are inappropriate. Simulated annealing uses an independent random-number stream so its sampling does not change subsequent network updates.

### Results and caveats

The [IEEE paper](https://doi.org/10.1109/SPERT67079.2025.11469727) reports experiments on its experimental configuration. This repository does **not** include a verified run log or a pinned environment proving those figures from the current source; do not quote its paper results as new outputs of the current corrected benchmark. Execute the notebook and preserve output, seeds, environment, and commit SHA to report new figures.

Additional limitations worth considering:

- Graph topologies and dynamics are *simulated*, without empirical interference measurements, radio propagation, or deployment conditions.
- Training labels derive in part from heuristic solutions, and repair influences final validity and switching cost.
- A fixed 50-channel output space is used by default and may be inappropriate for larger/chromatic-harder graphs.
- The model is evaluated on held-out **instances** of synthetic graph families, not on wholly unseen graph-generating distributions.
- Benchmarks use a small number of repeated update sequences; no confidence intervals or multi-seed sensitivity analysis are reported here.
- Algorithmic fairness may require equal quality constraints, wall-clock budgets, realistic update workloads, and ablation studies of neural inference versus repair alone.

## Repository map

| Path | Purpose |
| --- | --- |
| [`dynamic_gnn.py`](dynamic_gnn.py) | Model, graph features, incremental processing, color repair, baseline methods, graph generators, benchmark |
| [`Incremental_GNN_Combinatorial_Optimization.ipynb`](Incremental_GNN_Combinatorial_Optimization.ipynb) | Train and benchmark the model |
| [`tests/test_dynamic_gnn.py`](tests/test_dynamic_gnn.py) | Regression checks for conflicts, relabeling, update sampling, inference, benchmarks |
| [`.github/workflows/smoke-test.yml`](.github/workflows/smoke-test.yml) | Automated CPU smoke-test CI |
| [`requirements.txt`](requirements.txt) | Dependency compatibility ranges |

## Publication and citation

Pavan Parola, **“Incremental Graph Neural Networks for Dynamic Channel Assignment in HAN and NAN Deployments,”** *IEEE SPERT 2025*. DOI: [10.1109/SPERT67079.2025.11469727](https://doi.org/10.1109/SPERT67079.2025.11469727).

For archival citation details, use the metadata on the IEEE Xplore record. For implementation reproducibility, identify the **specific Git commit** as well as this paper.

## Scope

Research prototype, not production radio-network software or a certified channel-allocation system. The repository includes no real utility/customer data.
