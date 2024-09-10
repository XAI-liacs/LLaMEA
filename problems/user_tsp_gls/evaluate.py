import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prob import TSPGLS
import pickle
import time
import importlib
import os
import numpy as np
import tqdm

debug_mode = False
# problem_size = [10,20,50,100,200]
problem_size = [20,50,100]
n_test_ins = 1000

# Initialize DataFrame to collect gaps
gap_data = pd.DataFrame(columns=["Algorithm", "Problem_Size", "Gap"])

# Function to append gaps to DataFrame
def append_gaps_to_dataframe(algorithm, size, gaps, gap_data):
    for gap in gaps:
        gap_data.loc[len(gap_data)] = {"Algorithm": algorithm, "Problem_Size": size, "Gap": gap}
    return gap_data



goodalgs = ["EoH1", "EoH2", "EoH3", 
    #"AdaptiveVariationalPenaltyTSP", 
    #"DynamicAdaptivePenaltyTSP", "EnhancedDynamicEdgePenaltyPlusTSP", 
    "HeuristicAugmentedPenaltiesTSP", "RefinedDynamicEdgeMemoryPenaltyTSP", "RefinedDynamicEdgePenaltyPlusTSP"
    ]

labels = ["EoH-1", "EoH-2", "EoH-3", "LLaMEA-HPO-1", "LLaMEA-HPO-2", "LLaMEA-HPO-3"]

i = 0
for alg in tqdm.tqdm(goodalgs):
    #print(filename)
    h = alg
    prob = TSPGLS()
    prob.n_inst_eva = n_test_ins
    #print("Start evaluation...", h)
    with open(f"results/{h}.txt", "w") as file:
        for size in problem_size:
            path = os.path.dirname(os.path.abspath(__file__))
            instance_file_name = path+'/TestingData/TSP' + str(size)+ '.pkl'
            with open(instance_file_name, 'rb') as f:
                instance_dataset = pickle.load(f)

            # heuristic_module = importlib.import_module(f"algs.{h}")
            # heuristic = importlib.reload(heuristic_module)
            

            time_start = time.time()
            gaps = prob.testGLS(f"llamea.tspalgs.{h}", instance_dataset)

            result = (f"{h} - Average dis on {n_test_ins} instance with size {size} is: {np.mean(gaps):7.5f} timecost: {time.time()-time_start:7.3f}")
            #print(result)
            file.write(result + "\n")
            # Append gaps to DataFrame
            gap_data = append_gaps_to_dataframe(labels[i], size, gaps, gap_data) 
            i += 1


# for h in ["AdaptiveEdgePenaltyTSP", "DynamicPenaltyGradientOptimizer", "SynergisticAdaptivePenaltyRefinementPlus"]:

#     prob = TSPGLS()
#     prob.n_inst_eva = n_test_ins
#     print("Start evaluation...", h)
#     with open(f"results/{h}.txt", "w") as file:
#         for size in problem_size:
#             path = os.path.dirname(os.path.abspath(__file__))
#             instance_file_name = path+'/TestingData/TSP' + str(size)+ '.pkl'
#             with open(instance_file_name, 'rb') as f:
#                 instance_dataset = pickle.load(f)

#             heuristic_module = importlib.import_module(f"{h}")
#             heuristic = importlib.reload(heuristic_module)
            

#             time_start = time.time()
#             gaps = prob.testGLS(heuristic, instance_dataset)

#             result = (f"{h} - Average dis on {n_test_ins} instance with size {size} is: {np.mean(gaps):7.5f} timecost: {time.time()-time_start:7.3f}")
#             print(result)
#             file.write(result + "\n")
#             # Append gaps to DataFrame
#             gap_data = append_gaps_to_dataframe(h, size, gaps, gap_data)

# Save the DataFrame to a CSV and Pickle file
gap_data.to_csv("gap_data.csv", index=False)
gap_data.to_pickle("gap_data.pkl")

# Plot the violin plot using Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x="Problem_Size", y="Gap", hue="Algorithm", data=gap_data, split=True)
plt.xlabel("TSP Problem Size")
plt.title("Gap Distribution by Algorithm and Problem Size")
plt.savefig("tsp_results.pdf")