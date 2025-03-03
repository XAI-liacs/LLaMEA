# Uses Python Abstract Syntax Trees to extract graph characteristics
import ast
import difflib
import os
from collections import Counter

import jsonlines
import lizard
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, minmax_scale


def code_compare(code1, code2, printdiff=False):
    """
    Compares two Python code strings by computing a line-by-line diff and returns a similarity-based distance.

    Args:
        code1 (str): The first Python code snippet.
        code2 (str): The second Python code snippet.
        printdiff (bool): If True, prints the text diff (unused in current code).

    Returns:
        float: A value between 0 and 1 representing how dissimilar the two codes are.
        (0 means identical, 1 means completely different.)
    """

    # Parse the Python code into ASTs
    # Use difflib to find differences
    diff = difflib.ndiff(code1.splitlines(), code2.splitlines())
    # Count the number of differing lines
    diffs = sum(1 for x in diff if x.startswith("- ") or x.startswith("+ "))
    # Calculate total lines for the ratio
    total_lines = max(len(code1.splitlines()), len(code2.splitlines()))
    similarity_ratio = (total_lines - diffs) / total_lines if total_lines else 1
    return 1 - similarity_ratio


def analyse_complexity(code):
    """
    Analyzes the code complexity of a Python code snippet using the lizard library.

    Args:
        code (str): The Python code to analyze.

    Returns:
        dict: A dictionary containing statistics such as mean and total cyclomatic complexity,
        token counts, and parameter counts.
    """
    i = lizard.analyze_file.analyze_source_code("algorithm.py", code)
    complexities = []
    token_counts = []
    parameter_counts = []
    for f in i.function_list:
        complexities.append(f.__dict__["cyclomatic_complexity"])
        token_counts.append(f.__dict__["token_count"])
        parameter_counts.append(len(f.__dict__["full_parameters"]))
    return {
        "mean_complexity": np.mean(complexities),
        "total_complexity": np.sum(complexities),
        "mean_token_count": np.mean(token_counts),
        "total_token_count": np.sum(token_counts),
        "mean_parameter_count": np.mean(parameter_counts),
        "total_parameter_count": np.sum(parameter_counts),
    }


class BuildAST(ast.NodeVisitor):
    def __init__(self):
        """
        Class to build a directed graph representation (networkx.DiGraph) of a Python Abstract
        Syntax Tree (AST). Each AST node is represented as a graph node, and edges indicate
        parent-child relationships in the AST.
        """
        self.graph = nx.DiGraph()
        self.current_node = 0
        self.node_stack = []

    def generic_visit(self, node):
        """
        Visits each node in the AST. Adds the node to the graph, and connects it with an edge to
        its parent node. Uses a stack to keep track of the parent-child relationship.
        """
        node_id = self.current_node
        self.graph.add_node(node_id, label=type(node).__name__)

        if self.node_stack:
            parent_id = self.node_stack[-1]
            self.graph.add_edge(parent_id, node_id)

        self.node_stack.append(node_id)
        self.current_node += 1

        super().generic_visit(node)

        self.node_stack.pop()

    def build_graph(self, root):
        """
        Builds and returns the directed graph (networkx.DiGraph) from the AST root node by
        visiting each node in the tree.

        Args:
            root (ast.AST): The root of the AST from which to build the graph.

        Returns:
            networkx.DiGraph: A directed graph representing the AST.
        """
        self.visit(root)
        return self.graph


def eigenvector_centrality_numpy(G, max_iter=500):
    """
    Calculates the eigenvector centrality of a directed graph using networkx, returning
    NaN if it fails to compute (e.g., due to convergence issues).

    Args:
        G (networkx.DiGraph): The graph on which to compute centrality.
        max_iter (int): Maximum number of iterations for the eigenvector computation.

    Returns:
        tuple or float: A tuple containing a dictionary of centrality values or NaN if
        an exception occurs.
    """
    try:
        return (nx.eigenvector_centrality_numpy(G, max_iter=500),)
    except Exception:
        return np.nan


# Function to extract graph characteristics
def analyze_graph(G):
    """
    Analyzes a directed graph G and computes various graph characteristics, including
    the number of nodes/edges, degree statistics, transitivity, depth measures, clustering,
    and additional metrics like diameter, radius, and average shortest path.

    Args:
        G (networkx.DiGraph): The directed graph to analyze.

    Returns:
        dict: A dictionary of graph characteristics and statistics.
    """
    depths = dict(nx.single_source_shortest_path_length(G, min(G.nodes())))
    degrees = sorted((d for n, d in G.degree()), reverse=True)
    leaf_depths = [
        depth for node, depth in depths.items() if G.out_degree(node) == 0
    ]  # depth from root to leaves
    clustering_coefficients = list(nx.clustering(G).values())
    # Additional Features (not in paper)
    # Convert the directed graph to an undirected graph to avoid SCC problems
    undirected_G = G.to_undirected()
    if nx.is_connected(undirected_G):  # check if undirected graph is connected
        diameter = nx.diameter(undirected_G)
        radius = nx.radius(undirected_G)
        avg_shortest_path = nx.average_shortest_path_length(undirected_G)
        avg_eccentricity = np.mean(list(nx.eccentricity(undirected_G).values()))
    else:
        # Calculate diameter of the largest strongly connected component
        largest_cc = max(nx.connected_components(undirected_G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        radius = nx.radius(subgraph)
        avg_shortest_path = nx.average_shortest_path_length(subgraph)
        avg_eccentricity = np.mean(list(nx.eccentricity(subgraph).values()))
    edge_density = (
        G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1))
        if G.number_of_nodes() > 1
        else 0
    )

    return {
        # Number of Nodes and Edges
        "Nodes": G.number_of_nodes(),
        "Edges": G.number_of_edges(),
        # Degree Analysis
        # "Degrees": degrees,
        "Max Degree": max(degrees),
        "Min Degree": min(degrees),
        "Mean Degree": np.mean(degrees),
        "Degree Variance": np.var(degrees),
        # Transitivity
        "Transitivity": nx.transitivity(G),
        # Depth analysis
        # "Depths": leaf_depths,
        "Max Depth": max(leaf_depths),
        "Min Depth": min(leaf_depths),
        "Mean Depth": np.mean(leaf_depths),
        # Clustering Coefficients
        # "Clustering Coefficients": clustering_coefficients,
        "Max Clustering": max(clustering_coefficients),
        "Min Clustering": min(clustering_coefficients),
        "Mean Clustering": nx.average_clustering(G),
        "Clustering Variance": np.var(clustering_coefficients),
        # Entropy
        "Degree Entropy": entropy(degrees),
        "Depth Entropy": entropy(leaf_depths),
        # Additional features (not in paper)
        # "Betweenness Centrality": nx.betweenness_centrality(G),
        # "Eigenvector Centrality": eigenvector_centrality_numpy(G, max_iter=500),
        "Assortativity": nx.degree_assortativity_coefficient(G),
        "Average Eccentricity": avg_eccentricity,
        "Diameter": diameter,
        "Radius": radius,
        # "Pagerank": nx.pagerank(G, max_iter=500),
        "Edge Density": edge_density,
        "Average Shortest Path": avg_shortest_path,
    }


def visualize_graph(G):
    """
    Visualizes a directed graph using pydot/Graphviz. Draws node labels, edges, and saves
    the figure as 'graph1.pdf'.

    Args:
        G (networkx.DiGraph): The graph to visualize.
    """
    pos = graphviz_layout(G, prog="dot")
    labels = nx.get_node_attributes(G, "label")
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        arrows=True,
    )
    plt.savefig("graph1.pdf")


# Function to create graph out of AST
def process_file(path, visualize):
    """
    Processes a Python file from a given path by:
    1. Reading the code
    2. Parsing it into an AST
    3. Building a graph of the AST
    4. Analyzing the graph
    5. Optionally visualizing the graph
    6. Computing code complexity statistics

    Args:
        path (str): Path to the Python file.
        visualize (bool): If True, visualizes the graph and saves a PDF.

    Returns:
        dict: Combined statistics from graph analysis and code complexity.
    """
    with open(path, "r") as file:
        python_code = file.read()
    root = ast.parse(python_code)
    build = BuildAST()
    G = build.build_graph(root)
    stats = analyze_graph(G)

    complexity_stats = analyse_complexity(python_code)
    if visualize == True:  # visualize graph
        visualize_graph(G)
    return {**stats, **complexity_stats}


# Function to create graph out of AST
def process_code(python_code, visualize):
    """
    Processes a Python code string by:
    1. Parsing it into an AST
    2. Building a graph of the AST
    3. Analyzing the graph
    4. Optionally visualizing the graph
    5. Computing code complexity statistics

    Args:
        python_code (str): A string containing valid Python code.
        visualize (bool): If True, visualizes the resulting AST graph.

    Returns:
        dict: Combined statistics from graph analysis and code complexity.
    """
    root = ast.parse(python_code)
    build = BuildAST()
    G = build.build_graph(root)
    stats = analyze_graph(G)
    if visualize == True:  # visualize graph
        visualize_graph(G)

    complexity_stats = analyse_complexity(python_code)
    return {**stats, **complexity_stats}


def aggregate_stats(results):
    """
    Prints aggregate statistics across multiple graph analyses, such as total nodes,
    edges, transitivity, and clustering. Also prints max depth and average degree/edge
    density across all results.

    Args:
        results (list of dict): A list of dictionaries containing individual graph statistics.
    """
    print("Aggregate Statistics:")
    print("Total Nodes:", sum(result["Nodes"] for result in results))
    print("Total Edges:", sum(result["Edges"] for result in results))
    print(
        "Average Transitivity:",
        sum(result["Transitivity"] for result in results) / len(results),
    )
    print("Max Depth:", max(result["Max Depth"] for result in results))
    print(
        "Average Degree Mean:", np.mean([result["Mean Degree"] for result in results])
    )
    print(
        "Average Clustering Coefficient:",
        np.mean([result["Mean Clustering"] for result in results]),
    )
    print(
        "Average Eccentricity:",
        np.mean([result["Average Eccentricity"] for result in results]),
    )
    print(
        "Average Edge Density:", np.mean([result["Edge Density"] for result in results])
    )


def print_results(stats, file):
    """
    Prints detailed graph statistics for a single file or code snippet, including node
    and edge counts, degree information, clustering, entropy, centralities, and more.

    Args:
        stats (dict): A dictionary containing the graph and complexity stats.
        file (str): The name or identifier of the file (or code snippet) being analyzed.
    """
    print("Statistics for file:", file)
    print("Number of nodes:", stats["Nodes"])
    print("Number of Edges:", stats["Edges"])
    print("Degrees:", stats["Degrees"])
    print("Maximum Degree:", stats["Max Degree"])
    print("Minimum Degree:", stats["Min Degree"])
    print("Mean Degree:", stats["Mean Degree"])
    print("Degree Variance:", stats["Degree Variance"])
    print("Transitivity:", stats["Transitivity"])
    print("Leaf Depths:", stats["Depths"])
    print("Max Depth:", stats["Max Depth"])
    print("Min Depth:", stats["Min Depth"])
    print("Mean Depth:", stats["Mean Depth"])
    print("Clustering Coefficients:", stats["Clustering Coefficients"])
    print("Max Clustering:", stats["Max Clustering"])
    print("Min Clustering:", stats["Min Clustering"])
    print("Mean Clustering:", stats["Mean Clustering"])
    print("Clustering Variance:", stats["Clustering Variance"])
    print("Degree Entropy:", stats["Degree Entropy"])
    print("Depth Entropy:", stats["Depth Entropy"])
    print("Betweenness Centrality:", stats["Betweenness Centrality"])
    print("Eigenvector Centrality:", stats["Eigenvector Centrality"])
    print("Assortativity:", stats["Assortativity"])
    print("Average Eccentricity:", stats["Average Eccentricity"])
    print("Diameter:", stats["Diameter"])
    print("Radius:", stats["Radius"])
    print("Pagerank:", stats["Pagerank"])
    print("Edge Density:", stats["Edge Density"])
    print("Average Shortest Path:", stats["Average Shortest Path"])
    print("")


def process_file_paths(file_paths, visualize):
    """
    Processes multiple Python file paths by running 'process_file' on each. Optionally
    prints the results for each file and aggregates them.

    Args:
        file_paths (list): A list of file paths to Python scripts.
        visualize (bool): Whether to visualize each AST graph.
    """
    results = []
    for file_path in file_paths:
        stats = process_file(file_path, visualize)
        results.append(stats)
        print_results(stats, file_path)
    # aggregate_stats(results)


def analyze_run(
    expfolder, budget=100, label="LLaMEA", filename="ast.csv", visualize=True
):
    """
    Analyzes a single LLaMEA optimization run (e.g., a log of code solutions), computing
    AST-based metrics and code complexity for each generated solution. Logs the results
    to a CSV file and optionally generates visualizations of the run.

    Args:
        expfolder (str): The experiment folder containing a 'log.jsonl' file.
        budget (int): The maximum number of iterations or prompt count (unused in code).
        label (str): A label for identification in the resulting DataFrame.
        filename (str): Output CSV filename to store AST analysis results.
        visualize (bool): If True, calls 'plot_optimization_graphs' to visualize the run.
    """
    results = []
    alg_id = 0
    best_ever_fitness = -np.Inf

    log_file = f"{expfolder}/log.jsonl"
    print(log_file)
    if os.path.exists(log_file):
        with jsonlines.open(log_file) as reader:
            reader_i = -1
            for obj in reader.iter(type=dict, skip_invalid=True):
                reader_i += 1
                stats = []
                fitness = -np.Inf
                code = ""
                if "solution" in obj.keys():
                    code = obj["solution"]
                if "_solution" in obj.keys():  # Legacy log format
                    code = obj["_solution"]
                if "code" in obj.keys():  # EOH log file
                    code = obj["code"]
                if code == None:
                    continue
                if "parent_id" in obj.keys():
                    parents = [obj["parent_id"]]
                if "id" in obj.keys():
                    alg_id = obj["id"]
                if "parents" in obj.keys():
                    parents = obj["parents"]
                if (
                    "objective" in obj.keys() and obj["objective"] != None
                ):  # EOH log file
                    fitness = obj["objective"] * -1
                else:
                    fitness = -np.Inf
                try:
                    stats = process_code(code, False)
                except Exception as e:
                    print(e)
                    continue

                stats["fitness"] = 0.0
                stats["method"] = label
                stats["exp_dir"] = expfolder.replace("/", "_")
                stats["alg_id"] = alg_id
                stats["gen"] = reader_i
                stats["parent_ids"] = parents
                stats["fitness"] = fitness
                results.append(stats)
        resultdf = pd.DataFrame.from_dict(results)
        resultdf.to_csv(f"{expfolder}/{filename}", index=False)
        if visualize:
            plot_optimization_graphs(resultdf, expfolder)


def plot_optimization_graphs(data, expfolder):
    """
    Plots optimization progress and relationships between successive solutions in an
    evolutionary run based on AST metrics. Generates multiple plots showing how solutions
    change over time according to different projected axes (e.g. t-SNE, PCA).

    Args:
        data (pandas.DataFrame): DataFrame containing AST metrics, metadata, and fitness
            values for all solutions in the run.
        expfolder (str): Folder path where the generated plots are saved.
    """
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data["fitness"] = minmax_scale(data["fitness"])
    data.fillna(0, inplace=True)

    # Separate metadata and features
    metadata_cols = ["fitness", "method", "exp_dir", "alg_id", "parent_ids", "gen"]
    if "code_diff" in data.columns:
        metadata_cols.append("code_diff")
    features = data.drop(columns=metadata_cols)
    metadata = data[metadata_cols]

    # Convert string data to lists when needed
    data["parent_ids"] = data["parent_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Standardize features for PCA/tSNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Create a 1D projection using PCA
    pca = PCA(n_components=1)
    pca_projection = pca.fit_transform(features_scaled)
    data["pca_x"] = pca_projection[:, 0]

    # Create a 1D projection using t-SNE
    tsne = TSNE(n_components=1, random_state=42)
    tsne_projection = tsne.fit_transform(features_scaled)
    data["tsne_x"] = tsne_projection[:, 0]

    # Plot the evolution in t-SNE feature space
    parent_counts = Counter(
        parent_id for parent_ids in data["parent_ids"] for parent_id in parent_ids
    )

    data["parent_size"] = data["alg_id"].map(
        lambda x: (parent_counts[x]) if x in parent_counts else 1
    )

    for x_data in [
        "tsne_x",
        "pca_x",
        "total_complexity",
        "total_token_count",
        "total_parameter_count",
    ]:
        plt.figure()
        for _, row in data.iterrows():
            for parent_id in row["parent_ids"]:
                if parent_id in data["alg_id"].values:
                    parent_row = data[data["alg_id"] == parent_id].iloc[0]
                    plt.plot(
                        [parent_row["gen"], row["gen"]],
                        [parent_row[x_data], row[x_data]],
                        "-o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
                else:
                    plt.plot(
                        row["gen"],
                        row[x_data],
                        "o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
        plt.xlabel("Evaluation")
        plt.ylabel(x_data.replace("_", " "))
        plt.ylim(data[x_data].min() - 1, data[x_data].max() + 1)
        plt.tight_layout()
        plt.savefig(f"{expfolder}/{x_data}_Evolution.png")
        plt.close()
