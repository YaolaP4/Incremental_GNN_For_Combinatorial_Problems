# Incremental GNN for Dynamic Graph Coloring

This repository contains the current implementation of the incremental GNN used for dynamic graph-coloring experiments.

## Current implementation

The evaluation and coloring pipeline now:

- checks every unique undirected edge when computing conflicts;
- repairs residual conflicts and validates the final coloring before reporting results;
- uses an actual simulated-annealing baseline with a cooling schedule and Metropolis acceptance;
- evaluates on held-out synthetic graphs that are separate from the training set;
- uses the stated 70% edge-addition / 30% edge-removal dynamic update mix;
- lets every method maintain its own sequential coloring state;
- aligns arbitrary color labels produced by fresh global recolorings before counting channel changes;
- asserts that evaluated colorings are conflict-free instead of hard-coding a zero-conflict result.

The random-geometric evaluation graphs are synthetic spatial proxies. They are not real HAN/NAN field deployments.

## Run

Open `Incremental_GNN_Combinatorial_Optimization.ipynb` in Colab, Kaggle, or Jupyter and run it from top to bottom.

The notebook installs:

```text
torch
torch_geometric
networkx
numpy
scipy
tqdm
```

Training uses a two-phase schedule:

1. 20 epochs of static pre-training
2. 30 epochs of incremental training

The benchmark reports average computation time, conflicts, channel changes, and number of colors for:

- GNN
- DSATUR
- Welsh-Powell
- Greedy Repair
- Local Search SA

A small GitHub Actions smoke test runs on pushes to `main` and checks the real PyTorch Geometric forward/backward path, conflict repair, and an end-to-end benchmark update.

## Files

- `dynamic_gnn.py` — model, training, conflict repair, baselines, graph generation, and evaluation
- `Incremental_GNN_Combinatorial_Optimization.ipynb` — reproducible training/evaluation notebook
