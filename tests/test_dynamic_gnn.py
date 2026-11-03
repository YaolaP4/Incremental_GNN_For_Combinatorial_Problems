"""Regression checks for dynamic coloring and benchmark integrity.

Run with: python -m unittest discover -s tests -v
"""
import random
import unittest

import networkx as nx
import torch

from dynamic_gnn import (
    ConflictAwareDynamicGNN,
    DynamicGraphProcessor,
    align_color_labels,
    apply_updates,
    benchmark_graph_safe,
    count_conflicts,
    count_dict_conflicts,
    gnn_step,
    graph_to_edge_index,
    post_process_coloring,
    sample_updates,
    set_seed,
)


class ColoringTests(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_unique_edge_conflicts_and_repair(self):
        graph = nx.path_graph(5)
        edges = graph_to_edge_index(graph, torch.device("cpu"))
        prediction = torch.tensor([0, 1, 1, 0, 0])
        self.assertEqual(count_conflicts(edges, prediction), 2)
        result = post_process_coloring(prediction, edges, max_colors=5)
        self.assertEqual(count_conflicts(edges, result), 0)

    def test_color_relabeling_does_not_count_as_change(self):
        prev = {0: 0, 1: 1, 2: 0, 3: 2}
        renamed = {0: 7, 1: 4, 2: 7, 3: 9}
        self.assertEqual(align_color_labels(prev, renamed), prev)

    def test_empty_and_single_vertex_update_sampling(self):
        self.assertEqual(sample_updates(nx.empty_graph(), random.Random(1)), [])
        self.assertEqual(sample_updates(nx.empty_graph(1), random.Random(1)), [])
        self.assertEqual(sample_updates(nx.path_graph(3), random.Random(1), n_changes=0), [])
        with self.assertRaises(ValueError):
            sample_updates(nx.path_graph(3), random.Random(1), n_changes=-1)

    def test_inference_skips_teacher_target_and_returns_valid_coloring(self):
        graph = nx.path_graph(8)
        processor = DynamicGraphProcessor(max_colors=8, device=torch.device("cpu"))
        processor.process_static(graph)
        model = ConflictAwareDynamicGNN(max_colors=8, hidden=16).eval()
        previous = processor.last_coloring.copy()
        result, elapsed = gnn_step(
            model, processor,
            [{"type": "add_edge", "u": 0, "v": 7}],
            previous,
        )
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(count_dict_conflicts(processor.last_graph, result), 0)
        self.assertEqual(processor.last_coloring, result)

    def test_reproducible_graph_updates_independent_of_annealing(self):
        graph = nx.erdos_renyi_graph(20, 0.15, seed=10)
        rng_a = random.Random(12)
        rng_b = random.Random(12)
        a1 = sample_updates(graph, rng_a, n_changes=4)
        b1 = sample_updates(graph, rng_b, n_changes=4)
        self.assertEqual(a1, b1)
        next_graph = apply_updates(graph, a1)
        self.assertEqual(
            sample_updates(next_graph, rng_a, n_changes=4),
            sample_updates(next_graph, rng_b, n_changes=4),
        )

    def test_benchmark_all_methods_return_valid_colorings(self):
        model = ConflictAwareDynamicGNN(max_colors=12, hidden=16).eval()
        rows = benchmark_graph_safe(
            model, nx.erdos_renyi_graph(24, 0.1, seed=4),
            device=torch.device("cpu"), max_colors=12, update_rounds=2, seed=17,
        )
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["conflicts"] == 0 for row in rows))
        self.assertTrue(all(row["time_ms"] >= 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
