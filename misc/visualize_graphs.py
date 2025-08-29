import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from collections import Counter
import ast

# import xgboost as xgb

# Load the dataset without the "degrees" column
for problem in ["BP", "TSP", "BBO"]:
    data_path = f"ast/graphstats_{problem}.csv"
    fig_folder = f"ast/img{problem}/"
    os.makedirs(fig_folder, exist_ok=True)
    os.makedirs(f"{fig_folder}evo", exist_ok=True)

    # Read all columns except 'degrees'
    data = pd.read_csv(data_path)

    print(data["fitness"].describe())
    # data = data.drop(columns=["Betweenness Centrality"])

    # Replace NaN and infinite values with 0

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter out LLM configurations containing '1,1'
    data = data[~data["LLM"].str.contains("1,1")]
    data = data[~data["LLM"].str.contains("EoH")]

    plt.figure(figsize=(8, 6))
    plt.hist(data["fitness"], bins=20, edgecolor="black", alpha=0.7)
    plt.title("Histogram of LLaMEA Fitness")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"bins{problem}_fitness.png")
    plt.clf()

    if problem in ["BP", "TSP"]:
        # load also the EOH_BPO data and transform the original
        EOHpath = f"ast/graphstats_EOH_{problem}.csv"
        EOHdata = pd.read_csv(EOHpath)
        EOHdata.replace([np.inf, -np.inf], np.nan, inplace=True)
        EOHdata["fitness"] = EOHdata["fitness"]

        plt.figure(figsize=(8, 6))
        plt.hist(EOHdata["fitness"], bins=20, edgecolor="black", alpha=0.7)
        plt.title("Histogram of EoH Fitness")
        plt.xlabel("Fitness")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(f"EoH{problem}_fitness.png")
        plt.clf()

        # transform the data
        data["parent_ids"] = data["parent_id"].apply(
            lambda x: [x] if pd.notnull(x) else []
        )
        data["gen"] = data["alg_id"]
        data = data.drop(columns=["parent_id"])
        data = pd.concat([data, EOHdata], ignore_index=True)

    if problem == "BP":
        data["fitness"].fillna(-0.04, inplace=True)
        data.loc[data["fitness"] < -0.04, "fitness"] = -0.04
        # data["fitness"] = -1 * data["fitness"]
    elif problem == "BBO":
        data["fitness"].fillna(0, inplace=True)
    else:
        data["fitness"].fillna(-0.2, inplace=True)
        data.loc[data["fitness"] < -0.2, "fitness"] = -0.2
        # data["fitness"] = -1 * data["fitness"]
        # data["fitness"] = 1 + (data["fitness"] * 20)

    plt.figure(figsize=(8, 6))
    plt.hist(data["fitness"], bins=20, edgecolor="black", alpha=0.7)
    plt.title("Histogram of all Fitness")
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"all{problem}_fitness.png")
    plt.clf()

    print(problem, data["fitness"].describe())
    data["fitness"] = minmax_scale(data["fitness"])

    data.fillna(0, inplace=True)
    # data["fitness"] = minmax_scale(data["fitness"])

    print(data["fitness"].describe())

    # Separate metadata and features
    metadata_cols = ["fitness", "LLM", "exp_dir", "alg_id"]

    complexity_cols = [
        "mean_complexity",
        "total_complexity",
        "mean_token_count",
        "total_token_count",
        "mean_parameter_count",
        "total_parameter_count",
    ]

    if "code_diff" in data.columns:
        metadata_cols.append("code_diff")
    if problem in ["BP", "TSP"]:
        metadata_cols.append("gen")
        metadata_cols.append("parent_ids")
        data["parent_ids"] = data["parent_ids"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    else:
        metadata_cols.append("parent_id")
    features = data.drop(columns=metadata_cols)
    features = features.drop(columns=complexity_cols)
    metadata = data[metadata_cols]
    complexity_data = data[complexity_cols]

    # Standardize features for PCA/tSNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # print("Features:", len(features.columns), features.columns)

    # Create a 2D projection using PCA
    pca = PCA(n_components=2)
    pca_projection = pca.fit_transform(features_scaled)
    print(problem, "Explained variance ratio: ", pca.explained_variance_ratio_)
    print(problem, "PCA components", pca.components_)
    data["pca_x"], data["pca_y"] = pca_projection[:, 0], pca_projection[:, 1]

    # Create a 2D projection using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_projection = tsne.fit_transform(features_scaled)
    data["tsne_x"], data["tsne_y"] = tsne_projection[:, 0], tsne_projection[:, 1]

    # Plot PCA projection colored by fitness for each LLM
    for llm in metadata["LLM"].unique():
        subset = data[metadata["LLM"] == llm]
        plt.figure()
        plt.scatter(
            subset["pca_x"], subset["pca_y"], c=subset["fitness"], cmap="viridis", s=20
        )
        plt.colorbar(label="Fitness")
        plt.title(f"PCA Projection (Colored by Fitness) - LLM: {llm}")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.savefig(f"{fig_folder}PCA_Fitness_LLM_{llm}.png")
        plt.close()

    # Plot t-SNE projection colored by fitness for each LLM
    for llm in metadata["LLM"].unique():
        subset = data[metadata["LLM"] == llm]
        plt.figure()
        plt.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            c=subset["fitness"],
            cmap="viridis",
            s=20,
        )
        plt.colorbar(label="Fitness")
        plt.title(f"t-SNE Projection (Colored by Fitness) - LLM: {llm}")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(f"{fig_folder}tSNE_Fitness_LLM_{llm}.png")
        plt.close()

    # Plot PCA projection colored by LLM
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x="pca_x", y="pca_y", hue=metadata["LLM"], data=data, palette="tab10", s=20
    )
    plt.title("PCA Projection (Colored by LLM)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="LLM", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}{problem}_PCA_Projection_By_LLM.png")
    plt.close()

    # Plot t-SNE projection colored by LLM
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x="tsne_x", y="tsne_y", hue=metadata["LLM"], data=data, palette="tab10", s=20
    )
    plt.title("t-SNE Projection (Colored by LLM)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="LLM", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{fig_folder}tSNE_Projection_By_LLM.png")
    plt.close()

    # Plot the evolution of each graph feature over the optimization runs, split by LLM
    for llm in metadata["LLM"].unique():
        subset = data[metadata["LLM"] == llm]
        for feature in features.columns:
            plt.figure()
            exp_data_sorted = subset.sort_values(by="alg_id")
            plt.plot(exp_data_sorted["alg_id"], exp_data_sorted[feature])
            plt.title(f"Evolution of {feature} over Optimization Runs - {llm}")
            plt.xlabel("Evaluation")
            plt.ylabel(feature)
            plt.savefig(f"{fig_folder}Evolution_{feature}_LLM_{llm}.png")
            plt.close()

    # Plot t-SNE projection colored by experiment folder (exp_dir)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue=metadata["exp_dir"],
        data=data,
        palette="tab10",
        s=20,
        legend=False,
    )
    plt.title("t-SNE Projection (Colored by optimization run)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # plt.legend(title='Experiment Folder')
    plt.savefig(f"{fig_folder}tSNE_Projection_By_Exp_Folder.png")
    plt.close()

    # Create a 1D projection using PCA
    pca = PCA(n_components=1)
    pca_projection = pca.fit_transform(features_scaled)
    print(problem, "Explained variance 1D: ", pca.explained_variance_ratio_)
    data["pca_x"] = pca_projection[:, 0]

    # Create a 1D projection using t-SNE
    tsne = TSNE(n_components=1, random_state=42)
    tsne_projection = tsne.fit_transform(features_scaled)
    data["tsne_x"] = tsne_projection[:, 0]

    # Plot the evolution in t-SNE feature space for each experiment folder
    tsne_first_component = data["tsne_x"]
    for exp_dir in metadata["exp_dir"].unique():
        subset = data[data["exp_dir"] == exp_dir]

        if problem in ["BP", "TSP"]:
            parent_counts = Counter(
                parent_id
                for parent_ids in subset["parent_ids"]
                for parent_id in parent_ids
            )
        else:
            parent_counts = subset["parent_id"].value_counts()
        subset.loc[:, "parent_size"] = subset["alg_id"].map(
            lambda x: (parent_counts[x]) if x in parent_counts else 1
        )

        plt.figure()
        for _, row in subset.iterrows():
            if problem in ["BP", "TSP"]:
                for parent_id in row["parent_ids"]:
                    if parent_id in subset["alg_id"].values:
                        parent_row = subset[subset["alg_id"] == parent_id].iloc[0]
                        plt.plot(
                            [parent_row["gen"], row["gen"]],
                            [parent_row["tsne_x"], row["tsne_x"]],
                            "-",
                            color="k",
                            alpha=0.2,
                        )

                    plt.plot(
                        row["gen"],
                        row["tsne_x"],
                        "o",
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
            else:
                if row["parent_id"] in subset["alg_id"].values:
                    parent_row = subset[subset["alg_id"] == row["parent_id"]].iloc[0]
                    plt.plot(
                        [parent_row["alg_id"], row["alg_id"]],
                        [parent_row["tsne_x"], row["tsne_x"]],
                        "-",
                        color="k",
                        alpha=0.2,
                    )
                    plt.plot(
                        row["alg_id"],
                        row["tsne_x"],
                        "o",
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
        plt.title(f"Evolution in t-SNE Feature Space - Exp_dir: {exp_dir}")
        plt.xlabel("EvaluationD")
        plt.ylabel("t-SNE 1")
        plt.savefig(f'{fig_folder}evo/tSNE_Evolution_Exp_{exp_dir.replace("/","")}.png')
        plt.close()

    if problem not in ["BP", "TSP"]:
        # Plot the evolution in t-SNE feature space for each experiment folder, colored by fitness
        tsne_first_component = data["tsne_x"]
        for llm in metadata["LLM"].unique():
            llm_subset = data[data["LLM"] == llm]
            unique_exp_dirs = llm_subset["exp_dir"].unique()
            num_exp_dirs = len(unique_exp_dirs)
            num_cols = 3
            num_rows = int(np.ceil(num_exp_dirs / num_cols))
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(18, num_rows * 6), sharey=True
            )
            axes = axes.flatten()

            for i, exp_dir in enumerate(unique_exp_dirs):
                ax = axes[i]
                subset = llm_subset[llm_subset["exp_dir"] == exp_dir]
                if problem in ["BP", "TSP"]:
                    parent_counts = Counter(
                        parent_id
                        for parent_ids in subset["parent_ids"]
                        for parent_id in parent_ids
                    )
                else:
                    parent_counts = subset["parent_id"].value_counts()
                subset["parent_size"] = subset["alg_id"].map(
                    lambda x: parent_counts[x] + 5 if x in parent_counts else 5
                )

                for _, row in subset.iterrows():
                    if row["parent_id"] in subset["alg_id"].values:
                        parent_row = subset[subset["alg_id"] == row["parent_id"]].iloc[
                            0
                        ]
                        ax.plot(
                            [parent_row["alg_id"], row["alg_id"]],
                            [parent_row["tsne_x"], row["tsne_x"]],
                            "-o",
                            markersize=row["parent_size"],
                            color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                        )
                # ax.set_title(f'Exp_dir: {exp_dir}')
                ax.set_xlabel("Evaluation")
                ax.set_ylabel("t-SNE 1")
                ax.set_ylim(-80, 80)

            # Add colorbar as the last subplot
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=data["fitness"].min(), vmax=data["fitness"].max()
                ),
            )
            sm.set_array([])
            fig.colorbar(
                sm, ax=axes[-1], orientation="vertical", fraction=0.05, pad=0.05
            )

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle(f"Evolution in t-SNE Feature Space - {llm}", y=1.05)
            plt.tight_layout()
            plt.savefig(f"{fig_folder}evo/tSNE_Evolution_LLM_{llm}.png")
            plt.close()

    for code_feature in [
        "total_complexity",
        "total_token_count",
        "total_parameter_count",
    ]:
        # Plot the evolution in Cyclomatic Complexity space for each experiment folder, colored by fitness, with first 3 exp folders per LLM
        llms = metadata["LLM"].unique()
        print(llms)
        num_cols = 3
        num_rows = len(llms)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(18, num_rows * 6), sharey=True, sharex=False
        )
        axes = axes.reshape(num_rows, num_cols)

        for row_idx, llm in enumerate(llms):
            llm_subset = data[data["LLM"] == llm]
            unique_exp_dirs = llm_subset["exp_dir"].unique()[
                :3
            ]  # Only take the first 3 exp_dirs
            for col_idx, exp_dir in enumerate(unique_exp_dirs):
                ax = axes[row_idx, col_idx]
                subset = llm_subset[llm_subset["exp_dir"] == exp_dir]
                if problem in ["BP", "TSP"]:
                    parent_counts = Counter(
                        parent_id
                        for sublist in subset["parent_ids"]
                        for parent_id in sublist
                    )
                else:
                    parent_counts = subset["parent_id"].value_counts()
                subset["parent_size"] = subset["alg_id"].map(
                    lambda x: parent_counts[x] + 5 if x in parent_counts else 5
                )

                for _, row in subset.iterrows():
                    if problem in ["BP", "TSP"]:
                        for parent_id in row["parent_ids"]:
                            if parent_id in subset["alg_id"].values:
                                parent_row = subset[subset["alg_id"] == parent_id].iloc[
                                    0
                                ]
                                ax.plot(
                                    [parent_row["gen"], row["gen"]],
                                    [parent_row[code_feature], row[code_feature]],
                                    "-",
                                    color="k",
                                    alpha=0.2,
                                )

                            ax.plot(
                                row["gen"],
                                row[code_feature],
                                "o",
                                markersize=row["parent_size"],
                                color=plt.cm.viridis(
                                    row["fitness"] / max(data["fitness"])
                                ),
                            )

                    else:
                        if row["parent_id"] in subset["alg_id"].values:
                            parent_row = subset[
                                subset["alg_id"] == row["parent_id"]
                            ].iloc[0]
                            ax.plot(
                                [parent_row["alg_id"], row["alg_id"]],
                                [parent_row[code_feature], row[code_feature]],
                                "-",
                                color="k",
                                alpha=0.2,
                            )
                        ax.plot(
                            row["alg_id"],
                            row[code_feature],
                            "o",
                            markersize=row["parent_size"],
                            color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                        )
                ax.set_title(f"{llm}")  # , Exp_dir: {exp_dir}
                ax.set_xlabel("Evaluation")
                ax.set_ylabel(code_feature.replace("_", " "))
                ax.set_ylim(data[code_feature].min() - 1, data[code_feature].max() + 1)

        # Add colorbar as the last subplot in each row
        for row_idx in range(num_rows):
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=data["fitness"].min(), vmax=data["fitness"].max()
                ),
            )
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=axes[row_idx, -1],
                orientation="vertical",
                fraction=0.05,
                pad=0.05,
            )

        plt.suptitle(
            f'Evolution in {code_feature.replace("_", " ")} - All LLMs', y=1.02
        )
        plt.tight_layout()
        plt.savefig(f"{fig_folder}{problem}_{code_feature}_Evolution_All_LLMs.png")
        plt.close()

    # Plot the evolution in t-SNE feature space for each experiment folder, colored by fitness, with first 3 exp folders per LLM
    if problem not in ["BP", "TSP"]:
        llms = metadata["LLM"].unique()
        print(llms)
        num_cols = 3
        num_rows = len(llms)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(18, num_rows * 6), sharey=True, sharex=False
        )
        axes = axes.reshape(num_rows, num_cols)

        for row_idx, llm in enumerate(llms):
            llm_subset = data[data["LLM"] == llm]
            unique_exp_dirs = llm_subset["exp_dir"].unique()[
                :3
            ]  # Only take the first 3 exp_dirs
            for col_idx, exp_dir in enumerate(unique_exp_dirs):
                ax = axes[row_idx, col_idx]
                subset = llm_subset[llm_subset["exp_dir"] == exp_dir]
                if problem in ["BP", "TSP"]:
                    parent_counts = Counter(
                        parent_id
                        for parent_ids in subset["parent_ids"]
                        for parent_id in parent_ids
                    )
                else:
                    parent_counts = subset["parent_id"].value_counts()
                subset["parent_size"] = subset["alg_id"].map(
                    lambda x: parent_counts[x] + 5 if x in parent_counts else 5
                )

                for _, row in subset.iterrows():
                    if row["parent_id"] in subset["alg_id"].values:
                        parent_row = subset[subset["alg_id"] == row["parent_id"]].iloc[
                            0
                        ]
                        ax.plot(
                            [parent_row["alg_id"], row["alg_id"]],
                            [parent_row["tsne_x"], row["tsne_x"]],
                            "-o",
                            markersize=row["parent_size"],
                            color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                        )
                ax.set_title(f"{llm}")  # , Exp_dir: {exp_dir}
                ax.set_xlabel("Evaluation")
                ax.set_ylabel("t-SNE 1")
                ax.set_ylim(data["tsne_x"].min() - 5, data["tsne_x"].max() + 5)

        # Add colorbar as the last subplot in each row
        for row_idx in range(num_rows):
            sm = plt.cm.ScalarMappable(
                cmap="viridis",
                norm=plt.Normalize(
                    vmin=data["fitness"].min(), vmax=data["fitness"].max()
                ),
            )
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=axes[row_idx, -1],
                orientation="vertical",
                fraction=0.05,
                pad=0.05,
            )

        plt.suptitle("Evolution in t-SNE Feature Space - All LLMs", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{fig_folder}tSNE_Evolution_All_LLMs.png")
        plt.close()

    # Plot the evolution in PCA feature space for each experiment folder, colored by fitness, with first 3 exp folders per LLM
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(18, num_rows * 6), sharey=True, sharex=False
    )
    axes = axes.reshape(num_rows, num_cols)

    for row_idx, llm in enumerate(llms):
        llm_subset = data[data["LLM"] == llm]
        unique_exp_dirs = llm_subset["exp_dir"].unique()[
            :3
        ]  # Only take the first 3 exp_dirs
        for col_idx, exp_dir in enumerate(unique_exp_dirs):
            ax = axes[row_idx, col_idx]
            subset = llm_subset[llm_subset["exp_dir"] == exp_dir]
            if problem in ["BP", "TSP"]:
                parent_counts = Counter(
                    parent_id
                    for parent_ids in subset["parent_ids"]
                    for parent_id in parent_ids
                )
            else:
                parent_counts = subset["parent_id"].value_counts()
            subset.loc[:, "parent_size"] = subset["alg_id"].map(
                lambda x: parent_counts[x] + 3 if x in parent_counts else 3
            )

            for _, row in subset.iterrows():
                if problem in ["BP", "TSP"]:
                    for parent_id in row["parent_ids"]:
                        if parent_id in subset["alg_id"].values:
                            parent_row = subset[subset["alg_id"] == parent_id].iloc[0]
                            ax.plot(
                                [parent_row["gen"], row["gen"]],
                                [parent_row["pca_x"], row["pca_x"]],
                                "-",
                                color="k",
                                alpha=0.2,
                            )
                        ax.plot(
                            row["gen"],
                            row["pca_x"],
                            "o",
                            markersize=row["parent_size"],
                            color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                        )
                else:
                    if row["parent_id"] in subset["alg_id"].values:
                        parent_row = subset[subset["alg_id"] == row["parent_id"]].iloc[
                            0
                        ]
                        ax.plot(
                            [parent_row["alg_id"], row["alg_id"]],
                            [parent_row["pca_x"], row["pca_x"]],
                            "-",
                            color="k",
                            alpha=0.2,
                        )

                    ax.plot(
                        row["alg_id"],
                        row["pca_x"],
                        "o",
                        markersize=row["parent_size"],
                        color=plt.cm.viridis(row["fitness"] / max(data["fitness"])),
                    )
            ax.set_title(f"{llm}")  # , Exp_dir: {exp_dir}
            ax.set_xlabel("Evaluation")
            ax.set_ylabel(f"PCA 1- {pca.explained_variance_ratio_[0]:0.2f}")
            ax.set_ylim(data["pca_x"].min() - 1, data["pca_x"].max() + 1)

    # Add colorbar as the last subplot in each row
    for row_idx in range(num_rows):
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=data["fitness"].min(), vmax=data["fitness"].max()),
        )
        sm.set_array([])
        fig.colorbar(
            sm, ax=axes[row_idx, -1], orientation="vertical", fraction=0.05, pad=0.05
        )

    plt.suptitle("Evolution in PCA Feature Space - All LLMs", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{fig_folder}{problem}_PCA_Evolution_All_LLMs.png")
    plt.close()

    if True:
        # Create output directory if not exists
        output_dir = f"{fig_folder}features"
        os.makedirs(output_dir, exist_ok=True)

        # Correlation plot of each feature with fitness
        correlations = features.corrwith(metadata["fitness"])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.index, y=correlations.values)
        plt.xticks(rotation=90)
        plt.title("Correlation of Features with Fitness")
        plt.ylabel("Correlation Coefficient")
        plt.tight_layout()
        plt.savefig(f"{fig_folder}features/Feature_Fitness_Correlation.png")
        plt.close()

        # Train a Random Forest model and analyze feature importances
        X_train, X_test, y_train, y_test = train_test_split(
            features, metadata["fitness"], test_size=0.3, random_state=42
        )
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2_rf = rf.score(X_test, y_test)
        print(
            problem, f"Random Forest Model Performance: MSE: {mse:.4f} R^2: {r2_rf:.4f}"
        )

        # Plot feature importances
        importances = rf.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=features.columns, y=importances)
        plt.xticks(rotation=90)
        plt.title("Random Forest Feature Importances")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(f"{fig_folder}features/Random_Forest_Feature_Importance.png")
        plt.close()

        best_model = (
            rf  # max([(rf, r2_rf), (gb, r2_gb), (xgbr, r2_xgb)], key=lambda x: x[1])[0]
        )
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test, check_additivity=False)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(f"{fig_folder}features/SHAP_Feature_Importance.png")
        plt.close()
