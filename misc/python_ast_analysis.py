# Uses Python Abstract Syntax Trees to extract graph characteristics
import argparse
import ast
import difflib
import json

import jellyfish
import jsonlines
import lizard
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.stats import entropy


def code_compare(code1, code2, printdiff=False):
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
    # Analyse the code complexity of the code
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


# Parse Python AST and build a graph
class BuildAST(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.current_node = 0
        self.node_stack = []

    def generic_visit(self, node):
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
        self.visit(root)
        return self.graph


def eigenvector_centrality_numpy(G, max_iter=500):
    try:
        return (nx.eigenvector_centrality_numpy(G, max_iter=500),)
    except Exception:
        return np.nan


# Function to extract graph characteristics
def analyze_graph(G):
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
    root = ast.parse(python_code)
    build = BuildAST()
    G = build.build_graph(root)
    stats = analyze_graph(G)
    if visualize == True:  # visualize graph
        visualize_graph(G)

    complexity_stats = analyse_complexity(python_code)
    # print(stats)
    # print(complexity_stats)
    # print({**stats, **complexity_stats})
    # a = aaaa
    return {**stats, **complexity_stats}


def aggregate_stats(results):
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


def main(file_paths, visualize):
    results = []
    for file_path in file_paths:
        stats = process_file(file_path, visualize)
        results.append(stats)
        print_results(stats, file_path)
    # aggregate_stats(results)


if __name__ == "__main__":
    import os
    import sys

    budget = 100

    if True:  # switch to turn off some part of the code
        exp_dirs = [
            "runs/exp-09-06_122145-gpt-4o-2024-05-13-ES BP-HPO-long",
            "runs/exp-09-02_095606-gpt-4o-2024-05-13-ES BP-HPO-long",
            "runs/exp-08-30_141720-gpt-4o-2024-05-13-ES BP-HPO-long",
        ]

        colors = ["C0", "C0", "C1", "C1", "C2", "C2", "C3", "k"]
        linestyles = [
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "solid",
        ]
        labels = ["LLaMEA-HPO"]

        best_ever_code = ""
        best_ever_name = ""
        best_ever_config = {}
        best_ever_fitness = -100
        results = []
        for i in range(len(exp_dirs)):
            convergence = np.ones(budget) * -100
            # convergence_default = np.zeros(budget)
            code_diff_ratios = np.zeros(budget)
            best_so_far = -np.Inf
            best_so_far_default = 0
            previous_code = ""
            previous_config = {}
            previous_name = ""
            previous_gen = 0
            log_file = exp_dirs[i] + "/log.jsonl"
            if os.path.exists(log_file):
                with jsonlines.open(log_file) as reader:
                    for obj in reader.iter(type=dict, skip_invalid=True):
                        stats = []
                        gen = 0
                        fitness = None
                        code_diff = 0
                        code = ""
                        if "_solution" in obj.keys():
                            code = obj["_solution"]
                        if "_generation" in obj.keys():
                            gen = obj["_generation"]
                        if "_fitness" in obj.keys():
                            fitness = obj["_fitness"]
                            # print(fitness)
                        else:
                            fitness = None

                        if fitness > best_ever_fitness:
                            best_ever_fitness = fitness
                            best_ever_code = code
                            best_ever_config = obj["incumbent"]
                            best_ever_name = obj["_name"]

                        if fitness <= best_so_far:
                            code_diff = code_compare(previous_code, code, False)
                        else:
                            name = obj["_name"]
                            print(f"-- {gen} -- {previous_name} --> {name}")
                            code_diff = code_compare(previous_code, code, False)
                            best_so_far = fitness

                            previous_gen = gen
                            previous_code = code
                            previous_name = name
                            previous_config = obj["incumbent"]

                        code_diff_ratios[gen] = code_diff
                        convergence[gen] = fitness

                        try:
                            stats = process_code(code, False)
                        except Exception as e:
                            print(e)
                            continue

                        stats["code_diff"] = code_diff
                        stats["fitness"] = 0.0
                        stats["LLM"] = labels[0]
                        stats["exp_dir"] = exp_dirs[i]
                        stats["alg_id"] = gen
                        stats["parent_id"] = previous_gen
                        stats["fitness"] = fitness
                        results.append(stats)

        exp_dirs = ["run1", "run2", "run3"]
        convergence_lines = []
        for exp_dir in exp_dirs:
            conv_line = np.ones(budget * 200) * -np.Inf
            best_so_far = -np.Inf
            teller = 0
            gen = 0
            for k in range(5):
                with open(
                    "benchmarks/EoHresults/Prob1_OnlineBinPacking/"
                    + exp_dir
                    + f"/population_generation_{k}.json"
                ) as f:
                    pop = json.load(f)
                for ind in pop:
                    # if teller > budget:
                    #     break
                    if -1 * ind["objective"] > best_so_far:
                        best_so_far = -1 * ind["objective"]
                    conv_line[teller] = best_so_far
                    if k == 0:
                        teller += 1
                    else:
                        for x in range(5):  # EoH creates 5 offspring per individual
                            conv_line[teller] = best_so_far
                            teller += 1
                    code = ind["code"]
                    try:
                        stats = process_code(code, False)
                    except Exception:
                        continue

                    stats["code_diff"] = 0
                    stats["fitness"] = 0.0
                    stats["LLM"] = "EoH"
                    stats["exp_dir"] = exp_dir
                    stats["alg_id"] = gen
                    stats["parent_id"] = gen
                    stats["fitness"] = -1 * ind["objective"]
                    results.append(stats)
                    gen += 1

        convergence_lines.append(np.array(conv_line))

        import pandas as pd

        resultdf = pd.DataFrame.from_dict(results)
        print(resultdf)

        resultdf.to_csv("ast/graphstats_BP.csv", index=False)

        # TSP ------------------------------------------------------------

        exp_dirs = [
            "runs/exp-08-29_201655-gpt-4o-2024-05-13-ES TSP-HPO",
            "runs/exp-08-30_142330-gpt-4o-2024-05-13-ES TSP-HPO-deter",
            "runs/exp-09-02_105043-gpt-4o-2024-05-13-ES TSP-HPO-deter",
        ]

        colors = ["C0", "C0", "C1", "C1", "C2", "C2", "C3", "k"]
        linestyles = [
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "solid",
        ]
        labels = ["LLaMEA-HPO"]

        best_ever_code = ""
        best_ever_name = ""
        best_ever_config = {}
        best_ever_fitness = -100
        results = []
        for i in range(len(exp_dirs)):
            convergence = np.ones(budget) * -100
            # convergence_default = np.zeros(budget)
            code_diff_ratios = np.zeros(budget)
            best_so_far = -np.Inf
            best_so_far_default = 0
            previous_code = ""
            previous_config = {}
            previous_name = ""
            previous_gen = 0
            log_file = exp_dirs[i] + "/log.jsonl"
            if os.path.exists(log_file):
                with jsonlines.open(log_file) as reader:
                    for obj in reader.iter(type=dict, skip_invalid=True):
                        stats = []
                        gen = 0
                        fitness = None
                        code_diff = 0
                        code = ""
                        if "_solution" in obj.keys():
                            code = obj["_solution"]
                        if "_generation" in obj.keys():
                            gen = obj["_generation"]
                        if "_fitness" in obj.keys():
                            fitness = obj["_fitness"]
                            # print(fitness)
                        else:
                            fitness = None

                        if fitness > best_ever_fitness:
                            best_ever_fitness = fitness
                            best_ever_code = code
                            best_ever_config = obj["incumbent"]
                            best_ever_name = obj["_name"]

                        if fitness <= best_so_far:
                            code_diff = code_compare(previous_code, code, False)
                        else:
                            name = obj["_name"]
                            print(f"-- {gen} -- {previous_name} --> {name}")
                            code_diff = code_compare(previous_code, code, False)
                            best_so_far = fitness

                            previous_gen = gen
                            previous_code = code
                            previous_name = name
                            previous_config = obj["incumbent"]

                        code_diff_ratios[gen] = code_diff
                        convergence[gen] = fitness

                        try:
                            stats = process_code(code, False)
                        except Exception as e:
                            print(e)
                            continue

                        stats["code_diff"] = code_diff
                        stats["fitness"] = 0.0
                        stats["LLM"] = labels[0]
                        stats["exp_dir"] = exp_dirs[i]
                        stats["alg_id"] = gen
                        stats["parent_id"] = previous_gen
                        stats["fitness"] = fitness
                        results.append(stats)

        exp_dirs = ["run1", "run2", "run3"]
        convergence_lines = []
        for exp_dir in exp_dirs:
            conv_line = np.ones(budget * 200) * -np.Inf
            best_so_far = -np.Inf
            teller = 0
            gen = 0
            for k in range(5):
                with open(
                    "benchmarks/EoHresults/Prob2_TSP_GLS/"
                    + exp_dir
                    + f"/population_generation_{k}.json"
                ) as f:
                    pop = json.load(f)
                for ind in pop:
                    # if teller > budget:
                    #     break
                    if -1 * ind["objective"] > best_so_far:
                        best_so_far = -1 * ind["objective"]
                    conv_line[teller] = best_so_far
                    if k == 0:
                        teller += 1
                    else:
                        for x in range(5):  # EoH creates 5 offspring per individual
                            conv_line[teller] = best_so_far
                            teller += 1
                    code = ind["code"]
                    try:
                        stats = process_code(code, False)
                    except Exception:
                        continue

                    stats["code_diff"] = 0
                    stats["fitness"] = 0.0
                    stats["LLM"] = "EoH"
                    stats["exp_dir"] = exp_dir
                    stats["alg_id"] = gen
                    stats["parent_id"] = gen
                    stats["fitness"] = -1 * ind["objective"]
                    results.append(stats)
                    gen += 1

        convergence_lines.append(np.array(conv_line))

        import pandas as pd

        resultdf = pd.DataFrame.from_dict(results)
        print(resultdf)

        resultdf.to_csv("ast/graphstats_TSP.csv", index=False)

        # FIXED MUTATION RUNS ------------------------------

        base_dir = "ast/fixed_mutation_experiments/exp-prompt5-gpt-4o/"

        exp_dirs_2 = [
            "2/exp-11-01_150117-LLaMEA-gpt-4o-2",
            "2/exp-11-01_184846-LLaMEA-gpt-4o-2",
            "2/exp-11-01_211835-LLaMEA-gpt-4o-2",
        ]
        exp_dirs_5 = [
            "5/exp-11-01_150117-LLaMEA-gpt-4o-5",
            "5/exp-11-01_174133-LLaMEA-gpt-4o-5",
            "5/exp-11-01_192257-LLaMEA-gpt-4o-5",
        ]
        exp_dirs_10 = [
            "10/exp-11-01_150117-LLaMEA-gpt-4o-10",
            "10/exp-11-01_171429-LLaMEA-gpt-4o-10",
            "10/exp-11-01_201137-LLaMEA-gpt-4o-10",
        ]
        exp_dirs_20 = [
            "20/exp-11-01_150117-LLaMEA-gpt-4o-20",
            "20/exp-11-01_163509-LLaMEA-gpt-4o-20",
            "20/exp-11-01_200903-LLaMEA-gpt-4o-20",
        ]
        exp_dirs_40 = [
            "40/exp-11-01_150117-LLaMEA-gpt-4o-40",
            "40/exp-11-01_164611-LLaMEA-gpt-4o-40",
            "40/exp-11-01_184424-LLaMEA-gpt-4o-40",
        ]
        # TODO
        exp_dirs_var = [
            "LLaMEA/ast/dynamic_mutation_experiments/exp-gpt-4o-prompt5/exp-11-11_010046-LLaMEA-gpt-4o-beta1.5-exp1",
            "exp-11-11_025552-LLaMEA-gpt-4o-beta1.5-exp1",
            "exp-11-11_040455-LLaMEA-gpt-4o-beta1.5-exp1",
            "exp-11-11_064837-LLaMEA-gpt-4o-beta1.5-exp1",
            "exp-11-11_090052-LLaMEA-gpt-4o-beta1.5-exp1",
        ]

        colors = ["C0", "C0", "C1", "C1", "C2", "C2", "C3", "k"]
        linestyles = [
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "dotted",
            "solid",
            "solid",
        ]
        labels = ["LLaMEA-GPT4o"]
        exp_labels = ["2", "5", "10", "20", "40"]

        results = []
        for exp_i in range(
            len([exp_dirs_2, exp_dirs_5, exp_dirs_10, exp_dirs_20, exp_dirs_40])
        ):
            exp_dirs = [exp_dirs_2, exp_dirs_5, exp_dirs_10, exp_dirs_20, exp_dirs_40][
                exp_i
            ]
            exp_label = exp_labels[exp_i]

            for i in range(len(exp_dirs)):
                convergence = np.ones(budget) * -100
                # convergence_default = np.zeros(budget)
                code_diff_ratios = np.zeros(budget)
                best_so_far = -np.Inf
                best_so_far_default = 0
                previous_code = ""
                previous_config = {}
                previous_name = ""
                previous_gen = 0
                parent_id = 0
                best_so_far_id = 0
                alg_id = 0
                best_ever_code = ""
                best_ever_name = ""
                best_ever_config = {}
                best_ever_fitness = -100
                log_file = base_dir + exp_dirs[i] + "/log.jsonl"
                print(log_file)
                if os.path.exists(log_file):
                    with jsonlines.open(log_file) as reader:
                        reader_i = -1
                        for obj in reader.iter(type=dict, skip_invalid=True):
                            reader_i += 1
                            stats = []
                            gen = 0
                            fitness = None
                            code_diff = 0
                            code = ""
                            if "solution" in obj.keys():
                                code = obj["solution"]
                            if "generation" in obj.keys():
                                gen = obj["generation"]
                            # if "parent_id" in obj.keys():
                            parent_id = best_so_far_id
                            if "id" in obj.keys():
                                alg_id = obj["id"]
                            if "fitness" in obj.keys():
                                fitness = obj["fitness"]
                                # print(fitness)
                            else:
                                fitness = None

                            if fitness > best_ever_fitness:
                                best_ever_fitness = fitness
                                best_ever_code = code
                                best_ever_name = obj["name"]
                                best_so_far_id = reader_i

                            if fitness <= best_so_far:
                                code_diff = code_compare(previous_code, code, False)
                            else:
                                name = obj["name"]
                                print(f"-- {gen} -- {previous_name} --> {name}")
                                code_diff = code_compare(previous_code, code, False)
                                best_so_far = fitness

                                previous_gen = gen
                                previous_code = code
                                previous_name = name

                            code_diff_ratios[gen] = code_diff
                            convergence[gen] = fitness

                            try:
                                stats = process_code(code, False)
                            except Exception as e:
                                print(e)
                                continue

                            stats["code_diff"] = code_diff
                            stats["fitness"] = 0.0
                            stats["LLM"] = labels[0] + " " + exp_label
                            stats["exp_dir"] = exp_dirs[i].replace("/", "_")
                            stats["alg_id"] = reader_i
                            # stats["gen"] = reader_i
                            stats["parent_id"] = (
                                parent_id if parent_id != None else reader_i
                            )
                            stats["fitness"] = fitness
                            results.append(stats)

        resultdf = pd.DataFrame.from_dict(results)
        print(resultdf)

        resultdf.to_csv("ast/graphstats_BBO2.csv", index=False)

    # EOH runs ------------------------------
    import pandas as pd

    base_dir = "ast/EoH_runs/"

    exp_dirs_bpo = [
        "bpo_EoHrun1",
        "bpo_EoHrun2",
        "bpo_EoHrun3",
    ]
    exp_dirs_tsp = [
        "tsp_EoHrun1",
        "tsp_EoHrun2",
        "tsp_EoHrun3",
    ]

    colors = ["C0", "C0", "C0", "C1", "C1", "C1"]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "dotted",
        "dotted",
        "dotted",
    ]
    labels = ["EoH"]
    exp_labels = ["BPO", "TSP"]

    for exp_i in range(len([exp_dirs_bpo, exp_dirs_tsp])):
        results = []
        exp_dirs = [exp_dirs_bpo, exp_dirs_tsp][exp_i]
        exp_label = exp_labels[exp_i]

        for i in range(len(exp_dirs)):
            convergence = np.ones(budget) * -100
            # convergence_default = np.zeros(budget)
            code_diff_ratios = np.zeros(budget)
            best_so_far = -np.Inf
            best_so_far_default = 0
            previous_code = ""
            previous_config = {}
            previous_name = ""
            previous_gen = 0
            parent_id = 0
            best_so_far_id = 0
            alg_id = 0
            best_ever_code = ""
            best_ever_name = ""
            best_ever_config = {}
            best_ever_fitness = -100
            log_file = base_dir + exp_dirs[i] + ".jsonl"
            print(log_file)
            if os.path.exists(log_file):
                with jsonlines.open(log_file) as reader:
                    reader_i = -1
                    for obj in reader.iter(type=dict, skip_invalid=True):
                        reader_i += 1
                        stats = []
                        fitness = None
                        code_diff = 0
                        code = ""
                        if "code" in obj.keys():
                            code = obj["code"]
                        if code == None:
                            continue
                        # if "parent_id" in obj.keys():
                        parents = []
                        if "id" in obj.keys():
                            alg_id = obj["id"]
                        if "parents" in obj.keys():
                            parents = obj["parents"]
                        if "objective" in obj.keys() and obj["objective"] != None:
                            fitness = obj["objective"] * -1
                            # print(fitness)
                        else:
                            fitness = -100

                        if fitness > best_ever_fitness:
                            best_ever_fitness = fitness
                            best_ever_code = code
                            best_ever_name = obj["algorithm"]
                            best_so_far_id = reader_i

                        name = obj["algorithm"]
                        # print(f"-- {gen} -- {previous_name} --> {name}")
                        best_so_far = fitness
                        previous_code = code
                        previous_name = name

                        try:
                            stats = process_code(code, False)
                        except Exception as e:
                            print(e)
                            continue

                        stats["fitness"] = 0.0
                        stats["LLM"] = labels[0] + " " + exp_label
                        stats["exp_dir"] = exp_dirs[i].replace("/", "_")
                        stats["alg_id"] = alg_id
                        stats["gen"] = reader_i
                        stats["parent_ids"] = parents
                        stats["fitness"] = fitness
                        results.append(stats)

        resultdf = pd.DataFrame.from_dict(results)
        print(resultdf)

        resultdf.to_csv(f"ast/graphstats_EOH_{exp_label}.csv", index=False)
