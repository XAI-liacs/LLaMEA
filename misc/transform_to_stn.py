import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset without the "degrees" column

for problem in ["BP", "TSP", "BBO"]:
    data_path = f"ast/graphstats_{problem}.csv"

    # data_path = 'ast/graphstats.csv'
    data = pd.read_csv(data_path)

    # Drop irrelevant columns
    # data = data.drop(columns=["Degrees", "Pagerank", "Betweenness Centrality", "Clustering Coefficients", "Depths"])
    # Drop non-contributing columns
    data = data.drop(
        columns=[
            "Mean Clustering",
            "Max Clustering",
            "Min Depth",
            "Transitivity",
            "Clustering Variance",
            "mean_complexity",
            "total_complexity",
            "mean_token_count",
            "mean_parameter_count",
            "total_parameter_count",
        ]
    )  # "Eigenvector Centrality"

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # data["fitness"].fillna(-0.04, inplace=True)
    # data.loc[data["fitness"] < -0.04, 'fitness'] = -0.04

    # data["fitness"] = 1 + (data["fitness"] * 20)

    # data["fitness"] = minmax_scale(data["fitness"])

    data.fillna(0, inplace=True)

    # Separate metadata and features
    metadata_cols = ["fitness", "LLM", "exp_dir", "alg_id", "parent_id"]
    if "code_diff" in data.columns:
        metadata_cols.append("code_diff")
    features = data.drop(columns=metadata_cols)

    print(problem, features.columns, len(features.columns))

    metadata = data[metadata_cols]

    # Normalize feature columns
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Combine the normalized features with metadata
    data_scaled = pd.concat(
        [metadata.reset_index(drop=True), features_scaled_df], axis=1
    )

    # Create output directory if not exists
    output_dir = f"ast/{problem}_output_per_llm"
    os.makedirs(output_dir, exist_ok=True)

    # Process data for each LLM
    for llm in data_scaled["LLM"].unique():
        llm_data = data_scaled[data_scaled["LLM"] == llm]
        llm_output = []

        # Assign run numbers per exp_dir
        llm_data["run_number"] = llm_data.groupby("exp_dir").ngroup() + 1

        # Prepare the output
        for _, row in llm_data.iterrows():
            run_number = row["run_number"]
            fitness = row["fitness"]
            features_list = ",".join(map(str, row[features.columns].values))
            llm_output.append([run_number, fitness, features_list])

        # Create DataFrame for the output
        llm_output_df = pd.DataFrame(
            llm_output, columns=["run_number", "fitness", "features"]
        )

        # Sort by run_number
        llm_output_df = llm_output_df.sort_values(by="run_number")

        # Save to CSV
        output_file_path = os.path.join(output_dir, f"{llm}_output.csv")
        llm_output_df.to_csv(output_file_path, index=False, header=False, sep="\t")

    print(f"Output files saved in directory: {output_dir}")
