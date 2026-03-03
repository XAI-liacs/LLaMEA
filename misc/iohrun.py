"""Simple runner to analyze generated algorithms one by one on a wide benchmark set
"""
import numpy as np
from ioh import get_problem, logger, LogInfo
import re
import tqdm
from utils import OverBudgetException, budget_logger

auc_mean = 0
auc_std = 0
detailed_aucs = [0, 0, 0, 0, 0]

code_files = [
    "ioh/EnhancedFireworkAlgorithmWithAdaptiveLocalSearch.py",
    "ioh/ERADS_QuantumFluxUltraRefined.py",
    "ioh/ERADS_UltraDynamicMaxPlus.py",
    "ioh/ERADS_UltraDynamicPrecisionEnhanced.py",
    "ioh/QAPSO.py",
    "ioh/QPSO.py",
    "ioh/RADEA.py",
    "ioh/RefinedQuantumSwarmOptimizer.py",
    "ioh/QuantumDifferentialParticleOptimizerWithElitism.py",
    "ioh/EnhancedHybridCMAESDE.py",
    "ioh/ImprovedHybridCMAESDE.py",
    "ioh/AdaptiveHybridCMAESDE.py",
    "ioh/AdaptiveMemoryHybridDEPSO.py",
    "ioh/RefinedHybridDEPSOWithDynamicAdaptationV3.py",
    "ioh/AdaptiveHybridDEPSOWithDynamicRestart.py",
]

for code_file in code_files:
    # algorithm_name = re.findall("try-\d*-(\w*)\.py", code_file, re.IGNORECASE)[0]
    algorithm_name = re.findall("ioh/(\w*)\.py", code_file, re.IGNORECASE)[0]
    print("Benchmarking", algorithm_name)

    alg = ""
    # load alg to run
    with open(code_file, "r") as file:
        alg = file.read()
    # print(code_file, alg)

    exec(alg, globals())

    # store code in same folder
    # with open(f"ioh/{algorithm_name}.py", "w") as file:
    #    file.write(alg)

    for dim in [5, 10, 20]:
        budget = 2000 * dim

        loggers = [
            budget_logger(budget=budget, triggers=[logger.trigger.ALWAYS]),
            logger.Analyzer(
                folder_name=f"ioh/{algorithm_name}", algorithm_name=algorithm_name
            ),
        ]
        l1 = logger.Combine(loggers)
        for fid in tqdm.tqdm(np.arange(1, 25)):
            for iid in [1, 2, 3, 4, 5]:  # , 4, 5]
                problem = get_problem(fid, iid, dim)
                problem.attach_logger(l1)
                for rep in range(5):
                    np.random.seed(rep)
                    try:
                        algorithm = globals()[algorithm_name](budget)
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    except Exception:
                        pass
                    problem.reset()
