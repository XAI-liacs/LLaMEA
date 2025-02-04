# -------
# Evaluaiton code for EoH on Online Bin Packing
# --------
# More results may refer to
# Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang
# "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model"
# ICML 2024, https://arxiv.org/abs/2401.02051.

import importlib
from get_instance import GetData
from evaluation import Evaluation

eva = Evaluation()

capacity_list = [100, 200, 300, 400, 500]
# size_list = ['1k','5k','10k','100k']
size_list = ["1k", "5k", "10k"]  # ,'100k']

for h in ["EoH", "llamea"]:
    with open(f"results-{h}.txt", "w") as file:
        for capacity in capacity_list:
            for size in size_list:
                getdate = GetData()
                instances, lb = getdate.get_instances(capacity, size)
                heuristic_module = importlib.import_module(f"heuristic_{h}")
                heuristic = importlib.reload(heuristic_module)

                for name, dataset in instances.items():
                    avg_num_bins = -eva.evaluateGreedy(dataset, heuristic)
                    excess = (avg_num_bins - lb[name]) / lb[name]

                    result = f"{name}, {capacity}, Excess: {100 * excess:.2f}%"
                    print(h, result)
                    file.write(result + "\n")
