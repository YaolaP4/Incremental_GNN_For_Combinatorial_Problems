import copy
import math
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn


DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_to_edge_index(G, device):
    """Store each undirected edge exactly twice: forward block, then reverse block."""
    if G.number_of_edges() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ei = torch.tensor(list(G.edges()), dtype=torch.long, device=device).t().contiguous()
    return torch.cat([ei, ei[[1, 0]]], dim=1)


def unique_undirected_edges(edge_index):
    """Return each undirected edge exactly once, independent of storage ordering."""
    if edge_index.numel() == 0:
        return edge_index
    u, v = edge_index[0], edge_index[1]
    mask = u < v
    return edge_index[:, mask]


def count_conflicts(edge_index, colors):
    edges = unique_undirected_edges(edge_index)
    if edges.numel() == 0:
        return 0
    return int((colors[edges[0]] == colors[edges[1]]).sum().item())


def count_dict_conflicts(G, coloring):
    return sum(1 for u, v in G.edges() if coloring[u] == coloring[v])


def num_colors_dict(coloring):
    return len(set(coloring.values())) if coloring else 0


def num_color_changes(previous, current):
    return sum(1 for node, old_color in previous.items() if current.get(node) != old_color)


def tensor_to_coloring(nodes, colors):
    arr = colors.detach().cpu().tolist()
    return {node: int(arr[i]) for i, node in enumerate(nodes)}


class DynamicGraphProcessor:
    """Feature extraction and graph-state management for dynamic coloring."""

    def __init__(self, max_colors=50, device=None):
        self.max_colors = max_colors
        self.device = device or torch.device("cpu")
        self.base_feature_cache = None
        self.last_graph = None
        self.last_coloring = {}

    def compute_node_features(self, G, node):
        n = max(G.number_of_nodes(), 1)
        neighbors = list(G.neighbors(node))
        degree_norm = G.degree(node) / n

        if len(neighbors) >= 2:
            subgraph = G.subgraph(neighbors)
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering = subgraph.number_of_edges() / possible if possible else 0.0
        else:
            clustering = 0.0

        two_hop = set()
        for nbr in neighbors:
            two_hop.update(G.neighbors(nbr))
        two_hop.discard(node)
        two_hop_ratio = len(two_hop) / n

        scaled_degree = len(neighbors) / 20.0
        return [degree_norm, clustering, two_hop_ratio, scaled_degree]

    def _base_features(self, G, affected=None):
        nodes = list(G.nodes())
        n = len(nodes)
        if (
            self.base_feature_cache is None
            or self.base_feature_cache.shape[0] != n
            or affected is None
        ):
            feats = [self.compute_node_features(G, node) for node in nodes]
            return torch.tensor(feats, dtype=torch.float32, device=self.device)

        x = self.base_feature_cache.clone()
        for i, node in enumerate(nodes):
            if node in affected:
                x[i] = torch.tensor(
                    self.compute_node_features(G, node),
                    dtype=torch.float32,
                    device=self.device,
                )
        return x

    def _make_data(self, G, x_base, prev_coloring, y_coloring, affected=None):
        nodes = list(G.nodes())
        prev = torch.tensor(
            [prev_coloring.get(node, 0) for node in nodes],
            dtype=torch.long,
            device=self.device,
        )
        color_onehot = F.one_hot(prev, num_classes=self.max_colors).float()
        x = torch.cat([x_base, color_onehot], dim=1)
        y = torch.tensor(
            [y_coloring[node] for node in nodes],
            dtype=torch.long,
            device=self.device,
        )
        data = Data(
            x=x,
            edge_index=graph_to_edge_index(G, self.device),
            y=y,
            num_nodes=len(nodes),
        )
        data.nodes_list = nodes
        data.update_mask = torch.tensor(
            [node in (affected or set()) for node in nodes],
            dtype=torch.bool,
            device=self.device,
        )
        return data

    def process_static(self, G):
        G = nx.convert_node_labels_to_integers(G, ordering="default")
        nodes = list(G.nodes())
        teacher = nx.coloring.greedy_color(G, strategy="DSATUR")
        x_base = self._base_features(G, affected=None)

        x = torch.cat(
            [
                x_base,
                torch.zeros(
                    (len(nodes), self.max_colors),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        y = torch.tensor(
            [teacher[node] for node in nodes],
            dtype=torch.long,
            device=self.device,
        )
        data = Data(
            x=x,
            edge_index=graph_to_edge_index(G, self.device),
            y=y,
            num_nodes=len(nodes),
        )
        data.nodes_list = nodes
        data.update_mask = torch.zeros(
            len(nodes), dtype=torch.bool, device=self.device
        )

        self.base_feature_cache = x_base.clone()
        self.last_graph = G.copy()
        self.last_coloring = teacher.copy()
        return data

    def process_incremental(self, updates, prev_coloring=None):
        if self.last_graph is None:
            raise RuntimeError("Call process_static before process_incremental.")

        G = self.last_graph.copy()
        affected = set()

        for upd in updates:
            u, v = upd["u"], upd["v"]
            if upd["type"] == "add_edge":
                if u in G and v in G and u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    affected.update((u, v))
                    affected.update(G.neighbors(u))
                    affected.update(G.neighbors(v))
            elif upd["type"] == "remove_edge":
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    affected.update((u, v))
                    affected.update(G.neighbors(u))
                    affected.update(G.neighbors(v))
            else:
                raise ValueError(f"Unknown update type: {upd['type']}")

        previous = (prev_coloring or self.last_coloring).copy()
        teacher = greedy_repair_until_valid(G, previous, self.max_colors)
        x_base = self._base_features(G, affected)
        data = self._make_data(G, x_base, previous, teacher, affected)

        self.base_feature_cache = x_base.clone()
        self.last_graph = G.copy()
        self.last_coloring = teacher.copy()
        return data


class ConflictAwareDynamicGNN(nn.Module):
    """Two GraphSAGE layers, a two-head GAT layer, and a skip-connected color head."""

    def __init__(self, max_colors=50, hidden=64, gat_heads=2):
        super().__init__()
        in_feats = 4 + max_colors
        self.max_colors = max_colors
        self.sage1 = pyg_nn.SAGEConv(in_feats, hidden)
        self.sage2 = pyg_nn.SAGEConv(hidden, hidden)
        self.gat = pyg_nn.GATConv(hidden, hidden, heads=gat_heads, concat=False)
        self.color_head = nn.Sequential(
            nn.Linear(hidden + in_feats, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, max_colors),
        )
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, data):
        x_in = data.x
        x = F.relu(self.sage1(x_in, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.sage2(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat(x, data.edge_index)
        return self.color_head(torch.cat([x, x_in * self.skip_weight], dim=-1))


def conflict_aware_loss(
    logits,
    true_colors,
    edge_index,
    update_mask=None,
    conflict_weight=100.0,
):
    ce = F.cross_entropy(logits, true_colors, reduction="none")
    if update_mask is not None and bool(update_mask.any()):
        weights = torch.ones_like(ce)
        weights[update_mask] = 3.0
        ce = (ce * weights).mean()
    else:
        ce = ce.mean()

    edges = unique_undirected_edges(edge_index)
    if edges.numel() == 0:
        return ce

    probs = F.softmax(logits, dim=-1)
    u, v = edges[0], edges[1]
    soft_conflict = (probs[u] * probs[v]).sum(dim=1).mean()
    return ce + conflict_weight * soft_conflict


def post_process_coloring(pred_colors, edge_index, max_colors=50):
    """Single greedy pass over nodes, followed by an all-edge validity assertion."""
    colors = pred_colors.detach().cpu().clone().long()
    edges = unique_undirected_edges(edge_index).detach().cpu()
    if edges.numel() == 0:
        return colors.to(pred_colors.device)

    n = int(colors.numel())
    adj = [set() for _ in range(n)]
    for u, v in edges.t().tolist():
        adj[u].add(v)
        adj[v].add(u)

    for node in sorted(range(n), key=lambda x: len(adj[x]), reverse=True):
        used = {int(colors[w]) for w in adj[node]}
        if int(colors[node]) in used:
            replacement = next((c for c in range(max_colors) if c not in used), None)
            if replacement is None:
                raise ValueError("max_colors is too small for conflict repair.")
            colors[node] = replacement

    out = colors.to(pred_colors.device)
    remaining = count_conflicts(edge_index, out)
    if remaining != 0:
        raise AssertionError(f"Conflict repair left {remaining} conflicts.")
    return out


def greedy_repair_until_valid(G, coloring, max_colors=50):
    repaired = coloring.copy()
    for node in G.nodes():
        repaired.setdefault(node, 0)

    for _ in range(max(1, 2 * G.number_of_nodes())):
        conflicts = [(u, v) for u, v in G.edges() if repaired[u] == repaired[v]]
        if not conflicts:
            return repaired

        conflict_nodes = sorted(
            {n for edge in conflicts for n in edge},
            key=lambda n: G.degree(n),
            reverse=True,
        )
        changed = False
        for node in conflict_nodes:
            used = {repaired[nbr] for nbr in G.neighbors(node)}
            current = repaired[node]
            if current not in used:
                continue
            replacement = next((c for c in range(max_colors) if c not in used), None)
            if replacement is None:
                raise ValueError("max_colors is too small for greedy repair.")
            repaired[node] = replacement
            changed = True
        if not changed:
            break

    if count_dict_conflicts(G, repaired):
        raise RuntimeError("Greedy repair failed to produce a valid coloring.")
    return repaired


def align_color_labels(previous, current):
    """Align arbitrary fresh color IDs to prior channel IDs by maximum overlap."""
    old_labels = sorted(set(previous.values()))
    new_labels = sorted(set(current.values()))
    if not old_labels or not new_labels:
        return current.copy()

    overlap = np.zeros((len(new_labels), len(old_labels)), dtype=np.int64)
    new_idx = {c: i for i, c in enumerate(new_labels)}
    old_idx = {c: i for i, c in enumerate(old_labels)}
    for node, new_c in current.items():
        if node in previous:
            overlap[new_idx[new_c], old_idx[previous[node]]] += 1

    rows, cols = linear_sum_assignment(-overlap)
    mapping = {new_labels[r]: old_labels[c] for r, c in zip(rows, cols)}

    used_targets = set(old_labels)
    next_label = 0
    for c in new_labels:
        if c not in mapping:
            while next_label in used_targets:
                next_label += 1
            mapping[c] = next_label
            used_targets.add(next_label)

    return {node: mapping[c] for node, c in current.items()}


def dsatur_coloring(G, previous=None):
    coloring = nx.coloring.greedy_color(G, strategy="DSATUR")
    return align_color_labels(previous, coloring) if previous else coloring


def welsh_powell_coloring(G, previous=None):
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    return align_color_labels(previous, coloring) if previous else coloring


def annealing_objective(G, coloring, reference=None):
    conflicts = count_dict_conflicts(G, coloring)
    changes = num_color_changes(reference, coloring) if reference else 0
    colors = num_colors_dict(coloring)
    return 10_000 * conflicts + 10 * changes + colors


def simulated_annealing_coloring(
    G,
    init_coloring,
    max_colors=50,
    steps=2500,
    initial_temp=4.0,
    final_temp=0.02,
    rng=None,
):
    """Simulated annealing with cooling, Metropolis acceptance, and local-score deltas."""
    rng = rng or random.Random(DEFAULT_SEED)
    current = init_coloring.copy()
    for node in G.nodes():
        current.setdefault(node, 0)

    nodes = list(G.nodes())
    reference = init_coloring
    color_counts = defaultdict(int)
    for c in current.values():
        color_counts[c] += 1

    current_conflicts = count_dict_conflicts(G, current)
    current_changes = num_color_changes(reference, current)
    current_ncolors = sum(1 for n in color_counts.values() if n > 0)
    current_score = 10_000 * current_conflicts + 10 * current_changes + current_ncolors

    best = current.copy()
    best_score = current_score

    for step in range(max(1, steps)):
        frac = step / max(1, steps - 1)
        temp = initial_temp * ((final_temp / initial_temp) ** frac)

        conflicted = [
            n
            for n in nodes
            if any(current[n] == current[v] for v in G.neighbors(n))
        ]
        node = rng.choice(conflicted if conflicted and rng.random() < 0.85 else nodes)
        old_color = current[node]

        active_colors = sorted(c for c, count in color_counts.items() if count > 0)
        candidates = [c for c in active_colors if c != old_color]
        new_color = next((c for c in range(max_colors) if color_counts[c] == 0), None)
        if new_color is not None and new_color != old_color:
            candidates.append(new_color)
        if not candidates:
            continue

        used = {current[v] for v in G.neighbors(node)}
        valid = [c for c in candidates if c not in used]
        proposal = rng.choice(valid if valid and rng.random() < 0.9 else candidates)

        old_conflicts_at_node = sum(current[v] == old_color for v in G.neighbors(node))
        new_conflicts_at_node = sum(current[v] == proposal for v in G.neighbors(node))
        delta_conflicts = new_conflicts_at_node - old_conflicts_at_node

        old_changed = int(old_color != reference.get(node, old_color))
        new_changed = int(proposal != reference.get(node, old_color))
        delta_changes = new_changed - old_changed

        delta_colors = 0
        if color_counts[old_color] == 1:
            delta_colors -= 1
        if color_counts[proposal] == 0:
            delta_colors += 1

        delta = 10_000 * delta_conflicts + 10 * delta_changes + delta_colors

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current[node] = proposal
            color_counts[old_color] -= 1
            color_counts[proposal] += 1
            current_conflicts += delta_conflicts
            current_changes += delta_changes
            current_ncolors += delta_colors
            current_score += delta

            if current_score < best_score:
                best = current.copy()
                best_score = current_score

    return greedy_repair_until_valid(G, best, max_colors)


def sample_updates(G, rng, n_changes=None, add_probability=0.70):
    n_changes = n_changes or rng.randint(3, 5)
    work = G.copy()
    updates = []

    for _ in range(n_changes):
        want_add = rng.random() < add_probability
        if want_add:
            for _ in range(100):
                u, v = rng.sample(list(work.nodes()), 2)
                if u != v and not work.has_edge(u, v):
                    work.add_edge(u, v)
                    updates.append({"type": "add_edge", "u": u, "v": v})
                    break
        else:
            if work.number_of_edges() == 0:
                continue
            u, v = rng.choice(list(work.edges()))
            work.remove_edge(u, v)
            updates.append({"type": "remove_edge", "u": u, "v": v})

    return updates


def apply_updates(G, updates):
    G = G.copy()
    for upd in updates:
        u, v = upd["u"], upd["v"]
        if upd["type"] == "add_edge" and u != v:
            G.add_edge(u, v)
        elif upd["type"] == "remove_edge" and G.has_edge(u, v):
            G.remove_edge(u, v)
    return G


def generate_training_graphs(seed=1000):
    """50 synthetic training graphs plus two structured controls."""
    rng = random.Random(seed)
    graphs = []

    for _ in range(20):
        n = rng.randint(100, 500)
        p = rng.uniform(0.05, 0.15)
        graphs.append(nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9)))

    for _ in range(15):
        n = rng.randint(100, 500)
        m = rng.randint(2, 4)
        graphs.append(nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9)))

    for _ in range(15):
        n = rng.randint(100, 500)
        k = rng.choice([4, 6, 8])
        if k >= n:
            k = max(2, n - (1 - n % 2))
        graphs.append(nx.watts_strogatz_graph(n, k, 0.1, seed=rng.randint(0, 10**9)))

    controls = [nx.petersen_graph(), nx.grid_2d_graph(10, 10)]
    controls = [nx.convert_node_labels_to_integers(g) for g in controls]
    return graphs + controls


def generate_heldout_graphs(seed=9000):
    """Disjoint evaluation set; none of these graphs are used for training."""
    rng = random.Random(seed)
    medium = []
    for _ in range(3):
        medium.append(nx.erdos_renyi_graph(rng.randint(180, 320), rng.uniform(0.06, 0.11), seed=rng.randint(0, 10**9)))
        medium.append(nx.barabasi_albert_graph(rng.randint(180, 320), rng.randint(2, 4), seed=rng.randint(0, 10**9)))
        medium.append(nx.watts_strogatz_graph(rng.randint(180, 320), 6, 0.1, seed=rng.randint(0, 10**9)))

    realistic = [
        nx.random_geometric_graph(500, radius=0.085, seed=rng.randint(0, 10**9))
        for _ in range(3)
    ]

    large = [
        nx.erdos_renyi_graph(1000, 0.01, seed=rng.randint(0, 10**9)),
        nx.barabasi_albert_graph(1000, 4, seed=rng.randint(0, 10**9)),
        nx.watts_strogatz_graph(1000, 8, 0.1, seed=rng.randint(0, 10**9)),
    ]

    return {"medium": medium, "realistic": realistic, "large": large}


def train_static_model(model, processor, graphs, epochs=20, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        order = list(range(len(graphs)))
        random.shuffle(order)
        for idx in order:
            data = processor.process_static(graphs[idx])
            optimizer.zero_grad()
            logits = model(data)
            loss = conflict_aware_loss(
                logits,
                data.y,
                data.edge_index,
                conflict_weight=min(50.0 + 3.0 * epoch, 110.0),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())

        print(f"Static epoch {epoch + 1:02d}/{epochs}: loss={total / len(graphs):.4f}")

    return model


def train_incremental_model(model, processor, graphs, epochs=30, lr=1e-3, seed=2000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    rng = random.Random(seed)
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        total = 0.0
        order = list(range(len(graphs)))
        rng.shuffle(order)

        for idx in order:
            processor.process_static(graphs[idx])
            prev = processor.last_coloring.copy()

            for _ in range(rng.randint(3, 6)):
                updates = sample_updates(
                    processor.last_graph,
                    rng,
                    n_changes=rng.randint(3, 5),
                    add_probability=0.70,
                )
                if not updates:
                    continue

                data = processor.process_incremental(updates, prev_coloring=prev)
                optimizer.zero_grad()
                logits = model(data)
                conflict_weight = min(50.0 + 2.0 * epoch, 120.0)
                loss = conflict_aware_loss(
                    logits,
                    data.y,
                    data.edge_index,
                    data.update_mask,
                    conflict_weight=conflict_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                with torch.no_grad():
                    pred = post_process_coloring(
                        logits.argmax(dim=1),
                        data.edge_index,
                        processor.max_colors,
                    )
                prev = tensor_to_coloring(data.nodes_list, pred)
                processor.last_coloring = prev.copy()
                total += float(loss.item())

        scheduler.step(total)
        print(f"Incremental epoch {epoch + 1:02d}/{epochs}: loss={total:.4f}")
        if total < best_loss:
            best_loss = total
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def gnn_step(model, processor, updates, previous):
    data = processor.process_incremental(updates, prev_coloring=previous)
    start = time.perf_counter()
    logits = model(data)
    pred = post_process_coloring(logits.argmax(dim=1), data.edge_index, processor.max_colors)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    coloring = tensor_to_coloring(data.nodes_list, pred)
    processor.last_coloring = coloring.copy()
    return coloring, elapsed_ms


def benchmark_graph_safe(
    model,
    base_graph,
    device,
    max_colors=50,
    update_rounds=10,
    seed=0,
):
    rng = random.Random(seed)
    G = nx.convert_node_labels_to_integers(base_graph, ordering="default")

    common_initial = nx.coloring.greedy_color(G, strategy="DSATUR")
    states = {
        "GNN": common_initial.copy(),
        "DSATUR": common_initial.copy(),
        "Welsh-Powell": common_initial.copy(),
        "Greedy Repair": common_initial.copy(),
        "Local Search SA": common_initial.copy(),
    }

    processor = DynamicGraphProcessor(max_colors=max_colors, device=device)
    processor.process_static(G)
    processor.last_coloring = states["GNN"].copy()

    rows = []
    model.eval()

    for round_idx in range(update_rounds):
        updates = sample_updates(
            G,
            rng,
            n_changes=rng.randint(3, 5),
            add_probability=0.70,
        )
        if not updates:
            continue

        priors = {name: coloring.copy() for name, coloring in states.items()}
        G = apply_updates(G, updates)

        results = {}

        gnn_col, gnn_ms = gnn_step(model, processor, updates, priors["GNN"])
        states["GNN"] = gnn_col
        results["GNN"] = (gnn_col, gnn_ms)

        t0 = time.perf_counter()
        ds = dsatur_coloring(G, previous=priors["DSATUR"])
        results["DSATUR"] = (ds, (time.perf_counter() - t0) * 1000.0)
        states["DSATUR"] = ds

        t0 = time.perf_counter()
        wp = welsh_powell_coloring(G, previous=priors["Welsh-Powell"])
        results["Welsh-Powell"] = (wp, (time.perf_counter() - t0) * 1000.0)
        states["Welsh-Powell"] = wp

        t0 = time.perf_counter()
        gr = greedy_repair_until_valid(G, priors["Greedy Repair"], max_colors)
        results["Greedy Repair"] = (gr, (time.perf_counter() - t0) * 1000.0)
        states["Greedy Repair"] = gr

        t0 = time.perf_counter()
        sa = simulated_annealing_coloring(
            G,
            priors["Local Search SA"],
            max_colors=max_colors,
            steps=max(1000, min(4000, 4 * G.number_of_nodes())),
            rng=rng,
        )
        results["Local Search SA"] = (sa, (time.perf_counter() - t0) * 1000.0)
        states["Local Search SA"] = sa

        for method, (coloring, elapsed_ms) in results.items():
            conflicts = count_dict_conflicts(G, coloring)
            if conflicts != 0:
                raise AssertionError(f"{method} returned {conflicts} conflicts.")
            rows.append(
                {
                    "round": round_idx,
                    "method": method,
                    "time_ms": elapsed_ms,
                    "conflicts": conflicts,
                    "changes": num_color_changes(priors[method], coloring),
                    "colors": num_colors_dict(coloring),
                }
            )

    return rows


def summarize_rows(rows):
    methods = sorted({r["method"] for r in rows})
    summary = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {
            "time_ms": float(np.mean([r["time_ms"] for r in subset])),
            "conflicts": float(np.mean([r["conflicts"] for r in subset])),
            "changes": float(np.mean([r["changes"] for r in subset])),
            "colors": float(np.mean([r["colors"] for r in subset])),
        }
    return summary


def benchmark_suite(model, heldout, device, max_colors=50, rounds=10, seed=7000):
    output = {}
    for category, graphs in heldout.items():
        rows = []
        for i, G in enumerate(graphs):
            rows.extend(
                benchmark_graph_safe(
                    model,
                    G,
                    device=device,
                    max_colors=max_colors,
                    update_rounds=rounds,
                    seed=seed + 100 * i + len(rows),
                )
            )
        output[category] = {"rows": rows, "summary": summarize_rows(rows)}
    return output


def format_summary(summary):
    methods = ["GNN", "DSATUR", "Welsh-Powell", "Greedy Repair", "Local Search SA"]
    header = f"{'Method':<18} {'Time (ms)':>10} {'Conflicts':>10} {'Changes':>10} {'Colors':>10}"
    lines = [header, "-" * len(header)]
    for method in methods:
        if method not in summary:
            continue
        s = summary[method]
        lines.append(
            f"{method:<18} {s['time_ms']:>10.2f} {s['conflicts']:>10.2f} "
            f"{s['changes']:>10.2f} {s['colors']:>10.2f}"
        )
    return "\n".join(lines)
