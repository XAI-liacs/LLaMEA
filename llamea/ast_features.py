"""Utilities for extracting AST and code complexity features.

This module adapts the feature calculations used in
``misc.python_ast_analysis`` so they can be reused inside the LLaMEA
framework without pulling in the heavy visualization dependencies from the
original script.  The returned statistics mirror the ones that were already
available in the repository and combine structural information from the
abstract syntax tree with code complexity metrics obtained via ``lizard``.
"""

from __future__ import annotations

import ast
from typing import Dict, Iterable, List

import lizard
import networkx as nx
import numpy as np


class _ASTGraphBuilder(ast.NodeVisitor):
    """Build a directed graph representation of the AST."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._next_node = 0
        self._stack: List[int] = []

    def generic_visit(self, node: ast.AST) -> None:  # noqa: D401
        node_id = self._next_node
        self.graph.add_node(node_id, label=type(node).__name__)
        if self._stack:
            self.graph.add_edge(self._stack[-1], node_id)

        self._stack.append(node_id)
        self._next_node += 1

        super().generic_visit(node)

        self._stack.pop()

    def build(self, root: ast.AST) -> nx.DiGraph:
        self.visit(root)
        return self.graph


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _safe_sum(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.sum())


def _safe_var(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.var())


def _distribution_entropy(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    unique, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def _analyse_complexity(code: str) -> Dict[str, float]:
    analysis = lizard.analyze_file.analyze_source_code("algorithm.py", code)
    complexities: List[float] = []
    token_counts: List[float] = []
    parameter_counts: List[float] = []
    for function in analysis.function_list:
        complexities.append(getattr(function, "cyclomatic_complexity", 0.0))
        token_counts.append(getattr(function, "token_count", 0.0))
        parameter_counts.append(len(getattr(function, "full_parameters", [])))

    return {
        "mean_complexity": _safe_mean(complexities),
        "total_complexity": _safe_sum(complexities),
        "mean_token_count": _safe_mean(token_counts),
        "total_token_count": _safe_sum(token_counts),
        "mean_parameter_count": _safe_mean(parameter_counts),
        "total_parameter_count": _safe_sum(parameter_counts),
    }


def _analyze_graph(graph: nx.DiGraph) -> Dict[str, float]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]

    max_degree = float(max(degrees)) if degrees else 0.0
    min_degree = float(min(degrees)) if degrees else 0.0
    mean_degree = _safe_mean(degrees)
    degree_variance = _safe_var(degrees)

    try:
        transitivity = float(nx.transitivity(graph))
    except Exception:
        transitivity = 0.0

    if nodes:
        root = min(graph.nodes())
        path_lengths = dict(nx.single_source_shortest_path_length(graph, root))
    else:
        path_lengths = {}

    leaf_depths = [
        depth for node, depth in path_lengths.items() if graph.out_degree(node) == 0
    ]
    if not leaf_depths:
        leaf_depths = list(path_lengths.values())

    max_depth = float(max(leaf_depths)) if leaf_depths else 0.0
    min_depth = float(min(leaf_depths)) if leaf_depths else 0.0
    mean_depth = _safe_mean(leaf_depths)

    clustering_coeffs = list(nx.clustering(graph).values()) if nodes else []
    max_clustering = float(max(clustering_coeffs)) if clustering_coeffs else 0.0
    min_clustering = float(min(clustering_coeffs)) if clustering_coeffs else 0.0
    mean_clustering = _safe_mean(clustering_coeffs)
    clustering_variance = _safe_var(clustering_coeffs)

    degree_entropy = _distribution_entropy(degrees)
    depth_entropy = _distribution_entropy(leaf_depths)

    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
        if not np.isfinite(assortativity):
            assortativity = 0.0
    except Exception:
        assortativity = 0.0

    undirected = graph.to_undirected()
    if undirected.number_of_nodes() == 0:
        diameter = radius = avg_shortest_path = avg_eccentricity = 0.0
    else:
        if nx.is_connected(undirected):
            connected = undirected
        else:
            largest_component = max(
                nx.connected_components(undirected), key=len, default=None
            )
            connected = undirected.subgraph(largest_component).copy()

        try:
            diameter = float(nx.diameter(connected))
        except Exception:
            diameter = 0.0

        try:
            radius = float(nx.radius(connected))
        except Exception:
            radius = 0.0

        try:
            avg_shortest_path = float(nx.average_shortest_path_length(connected))
        except Exception:
            avg_shortest_path = 0.0

        try:
            ecc = nx.eccentricity(connected)
            avg_eccentricity = float(np.mean(list(ecc.values())))
        except Exception:
            avg_eccentricity = 0.0

    if nodes > 1:
        edge_density = float(edges / (nodes * (nodes - 1)))
    else:
        edge_density = 0.0

    return {
        "Nodes": float(nodes),
        "Edges": float(edges),
        "Max Degree": max_degree,
        "Min Degree": min_degree,
        "Mean Degree": mean_degree,
        "Degree Variance": degree_variance,
        "Transitivity": transitivity,
        "Max Depth": max_depth,
        "Min Depth": min_depth,
        "Mean Depth": mean_depth,
        "Max Clustering": max_clustering,
        "Min Clustering": min_clustering,
        "Mean Clustering": mean_clustering,
        "Clustering Variance": clustering_variance,
        "Degree Entropy": degree_entropy,
        "Depth Entropy": depth_entropy,
        "Assortativity": assortativity,
        "Average Eccentricity": avg_eccentricity,
        "Diameter": diameter,
        "Radius": radius,
        "Edge Density": edge_density,
        "Average Shortest Path": avg_shortest_path,
    }


def extract_ast_features(code: str) -> Dict[str, float]:
    """Return a dictionary with AST and complexity features for ``code``.

    Args:
        code: Python source code of the solution.

    Returns:
        A mapping from feature name to float value.

    Raises:
        ValueError: If the code cannot be parsed into an AST.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as err:  # pragma: no cover - propagated to caller
        raise ValueError("Failed to parse solution code") from err

    builder = _ASTGraphBuilder()
    graph = builder.build(tree)
    stats = _analyze_graph(graph)
    complexity = _analyse_complexity(code)

    features = {**stats, **complexity}
    return {name: float(value) for name, value in features.items()}


__all__ = ["extract_ast_features"]
