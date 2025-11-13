import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import textwrap
import re
import os
#from IPython.display import display, Markdown
from itertools import product
import itertools

output_files = [
        "exp-11-07_095145-LLaMEA-gpt-5-nano-ELA-Separable_GlobalLocal-sharing.jsonl",
        "exp-11-07_100349-LLaMEA-gpt-5-nano-ELA-Separable_Multimodality-sharing.jsonl",
        "exp-11-07_101329-LLaMEA-gpt-5-nano-ELA-Separable_Basins-sharing.jsonl",
        "exp-11-07_102426-LLaMEA-gpt-5-nano-ELA-Separable_Homogeneous-sharing.jsonl",
        "exp-11-07_103559-LLaMEA-gpt-5-nano-ELA-Separable-sharing.jsonl",
        "exp-11-07_104417-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Multimodality-sharing.jsonl",
        "exp-11-07_105439-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Basins-sharing.jsonl",
        "exp-11-07_131500-LLaMEA-gpt-5-nano-ELA-GlobalLocal_Homogeneous-sharing.jsonl",
        "exp-11-07_151340-LLaMEA-gpt-5-nano-ELA-GlobalLocal-sharing.jsonl",
        "exp-11-07_153338-LLaMEA-gpt-5-nano-ELA-Multimodality_Basins-sharing.jsonl",
        "exp-11-07_191623-LLaMEA-gpt-5-nano-ELA-Multimodality_Homogeneous-sharing.jsonl",
        "exp-11-07_192657-LLaMEA-gpt-5-nano-ELA-Multimodality-sharing.jsonl",
        "exp-11-08_121302-LLaMEA-gpt-5-nano-ELA-Basins_Homogeneous-sharing.jsonl",
        "exp-11-08_135452-LLaMEA-gpt-5-nano-ELA-Basins-sharing.jsonl",
        "exp-11-08_175315-LLaMEA-gpt-5-nano-ELA-Homogeneous-sharing.jsonl",
        "exp-11-08_182445-LLaMEA-gpt-5-nano-ELA-NOT Basins_Separable-sharing.jsonl",
        "exp-11-08_191508-LLaMEA-gpt-5-nano-ELA-NOT Basins_GlobalLocal-sharing.jsonl",
        "exp-11-08_195321-LLaMEA-gpt-5-nano-ELA-NOT Basins_Multimodality-sharing.jsonl",
        "exp-11-08_222921-LLaMEA-gpt-5-nano-ELA-NOT Basins-sharing.jsonl",
        "exp-11-08_230434-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Separable-sharing.jsonl",
        "exp-11-08_234910-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_GlobalLocal-sharing.jsonl",
        "exp-11-09_003403-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous_Multimodality-sharing.jsonl",
        "exp-11-09_010643-LLaMEA-gpt-5-nano-ELA-NOT Homogeneous-sharing.jsonl"
    ]



# --- SETTINGS ---
bounds = (-5, 5)
dim = 2  # visualize in 2D
grid_points = 200  # resolution


for f in output_files:
    file_path = f"landscapes/{f}" 

    filename = os.path.basename(file_path)
    match = re.search(r"-ELA-(.*?)(?:-|\-sharing\.jsonl$)", filename)
    title = match.group(1).replace("_", " ") if match else filename

    # --- DISPLAY MARKDOWN TITLE ---
    #display(Markdown(f"## {title}"))
    print(f"Processing file: {file_path}")

    # --- LOAD JSON LINES ---
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    n = len(data)
    cols = 8
    rows = math.ceil(n / cols)

    # --- SETUP FIGURE ---
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    # --- EVALUATE AND PLOT EACH LANDSCAPE ---
    # This is also an example of how you can load the Python code for each of the generated landscapes easily.
    x = np.linspace(*bounds, grid_points)
    y = np.linspace(*bounds, grid_points)
    X, Y = np.meshgrid(x, y)

    for i, entry in enumerate(data):
        ax = axes[i]
        code = entry["code"]
        try:
            # Execute class definition safely
            exec(code, globals())
            cls = globals()[entry["name"]]
            func = cls(dim=dim).f

            # Evaluate f on 2D grid
            Z = np.zeros_like(X)
            for ix in range(grid_points):
                for iy in range(grid_points):
                    Z[iy, ix] = func([X[iy, ix], Y[iy, ix]])

            # Plot
            #sns.kdeplot(x=x, y=y, fill=False, cmap="viridis", thresh=0, levels=100, ax=ax)
            #cs = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
            cs = ax.contourf(
                X, Y, Z,
                levels=50,
                cmap="inferno",
                alpha=0.75,   # partial transparency,
                antialiased=True
            )

            cl = ax.contour(
                X, Y, Z,
                levels=15,    # fewer, clean lines
                colors="black",
                linewidths=0.5,
                alpha=0.7,
                antialiased=True
            )
            ax.set_frame_on(False)
            #ax.set_title(textwrap.shorten(entry["description"], width=60, placeholder="…"), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    #fig.suptitle(f"2D Contour Landscapes - {title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"landscapes/{title.replace(' ', '_')}_landscapes.png")
    plt.close()