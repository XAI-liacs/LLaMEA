import os
import sys
import ioh
import time
import multiprocessing
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
sys.path.append(".")
from llamea import LLaMEA, llm
from llamea.utils import NoCodeException
from llamea.solution import Solution
from misc.utils import final_y_logger, OverBudgetException


# Define the GNBG class
class GNBG:
    def __init__(self, MaxEvals, AcceptanceThreshold, Dimension, CompNum,
                 MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu,
                 Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition):
        self.MaxEvals = MaxEvals
        self.AcceptanceThreshold = AcceptanceThreshold
        self.Dimension = Dimension
        self.CompNum = CompNum
        self.MinCoordinate = MinCoordinate
        self.MaxCoordinate = MaxCoordinate
        self.CompMinPos = CompMinPos
        self.CompSigma = CompSigma
        self.CompH = CompH
        self.Mu = Mu
        self.Omega = Omega
        self.Lambda = Lambda
        self.RotationMatrix = RotationMatrix
        self.OptimumValue = OptimumValue
        self.OptimumPosition = OptimumPosition
        self.FEhistory = []
        self.FE = 0
        self.AcceptanceReachPoint = np.inf
        self.BestFoundResult = np.inf

    def fitness(self, X):
        if len(X.shape) < 2:
            X = X.reshape(1, -1)
        SolutionNumber = X.shape[0]
        result = np.nan * np.ones(SolutionNumber)
        for jj in range(SolutionNumber):
            x = X[jj, :].reshape(-1, 1)  # Ensure column vector
            f = np.nan * np.ones(self.CompNum)
            for k in range(self.CompNum):
                if len(self.RotationMatrix.shape) == 3:
                    rotation_matrix = self.RotationMatrix[:, :, k]
                else:
                    rotation_matrix = self.RotationMatrix

                a = self.transform(
                    (x - self.CompMinPos[k, :].reshape(-1, 1)).T @ rotation_matrix.T, self.Mu[k, :], self.Omega[k, :])
                b = self.transform(
                    rotation_matrix @ (x - self.CompMinPos[k, :].reshape(-1, 1)), self.Mu[k, :], self.Omega[k, :])
                f[k] = self.CompSigma[k] + \
                    (a @ np.diag(self.CompH[k, :]) @ b) ** self.Lambda[k]

            result[jj] = np.min(f)
            if self.FE > (self.MaxEvals-1):
                return result
            self.FE += 1
            self.FEhistory = np.append(self.FEhistory, result[jj])
            if self.BestFoundResult > result[jj]:
                self.BestFoundResult = result[jj]
            if abs(self.FEhistory[self.FE-1] - self.OptimumValue) < self.AcceptanceThreshold and np.isinf(self.AcceptanceReachPoint):
                self.AcceptanceReachPoint = self.FE
        return result

    def transform(self, X, Alpha, Beta):
        Y = X.copy()
        tmp = (X > 0)
        Y[tmp] = np.log(X[tmp])
        Y[tmp] = np.exp(Y[tmp] + Alpha[0] *
                        (np.sin(Beta[0] * Y[tmp]) + np.sin(Beta[1] * Y[tmp])))
        tmp = (X < 0)
        Y[tmp] = np.log(-X[tmp])
        Y[tmp] = -np.exp(Y[tmp] + Alpha[1] *
                         (np.sin(Beta[2] * Y[tmp]) + np.sin(Beta[3] * Y[tmp])))
        return Y


def benchmark_alg(algorithm_name, problem, dim, budget, optimum):
    ioh.problem.wrap_real_problem(problem.fitness, name=f"GNBM",
                                    optimization_type=ioh.OptimizationType.MIN,
                                    lb=-100., ub=100.)
    ioh_problem = ioh.get_problem(f"GNBM", dimension=dim)
    np.random.seed()
    try:
        algorithm = globals()[algorithm_name](budget=budget, dim=dim)
        algorithm(ioh_problem)
    except OverBudgetException:
        pass
    ioh_problem.reset()
    return abs(ioh_problem.state.current_best.y-optimum)


def evaluateGNBG(solution, explogger=None):
    start_time = time.time()
    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())
    algorithm = None
    
    problems = []
    dims = []
    optimums = []
    bugets = [500000 for _ in range(15)] + [1000000 for _ in range(9)]
    for ProblemIndex in range(1, 2):
        print("problem", ProblemIndex, "loading")
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the path to the folder where you want to read/write files
        folder_path = os.path.join(current_dir)
        # Preparation and loading of the GNBG parameters based on the chosen problem instance
        if 1 <= ProblemIndex <= 24:
            filename = f'f{ProblemIndex}.mat'
            GNBG_tmp = loadmat(os.path.join(folder_path, filename))['GNBG']
            MaxEvals = np.array([item[0]
                                for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
            AcceptanceThreshold = np.array(
                [item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
            Dimension = np.array([item[0]
                                for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
            CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[
                0, 0]  # Number of components
            MinCoordinate = np.array(
                [item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
            MaxCoordinate = np.array(
                [item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
            CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
            CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
            CompH = np.array(GNBG_tmp['Component_H'][0, 0])
            Mu = np.array(GNBG_tmp['Mu'][0, 0])
            Omega = np.array(GNBG_tmp['Omega'][0, 0])
            Lambda = np.array(GNBG_tmp['lambda'][0, 0])
            RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
            OptimumValue = np.array(
                [item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
            OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
        else:
            raise ValueError('ProblemIndex must be between 1 and 24.')

        gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum,
                    MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH,
                    Mu, Omega, Lambda, RotationMatrix, OptimumValue,
                    OptimumPosition)
        problems += [gnbg]
        dims += [Dimension]
        optimums += [OptimumValue]
    print("problems loaded")
    final_y = []
    with multiprocessing.Pool(processes=24) as pool:
        final_y = pool.starmap(benchmark_alg, [(algorithm_name, problems[i], dims[i], bugets[i], optimums[i]) for i in range(1)])
    for i in range(len(problems)):
        problem = problems[i]
        dim = dims[i]
        budget = bugets[i]
        ioh.problem.wrap_real_problem(problem.fitness, name=f"GNBM_{i}",
                                      optimization_type=ioh.OptimizationType.MIN,
                                      lb=-100., ub=100.)
        ioh_problem = ioh.get_problem(f"GNBM_{i}", dimension=dim)
        l2 = final_y_logger(budget=budget, triggers=[ioh.logger.trigger.ALWAYS])
        ioh_problem.attach_logger(l2)
        for run in range(1):
            np.random.seed(run)
            try:
                algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                algorithm(ioh_problem)
            except OverBudgetException:
                pass
            final_y += [abs(ioh_problem.state.current_best.y-optimums[i])]
            print(final_y[-1])
            l2.reset(ioh_problem)
            ioh_problem.reset()


    feedback = f"The algorithm {algorithm_name} got a mean value of best solutions found was {np.mean(final_y):0.8f} (0. is the best) with standard deviation {np.std(final_y):0.8f}."

    print(feedback)
    solution.add_metadata("final_y", final_y)
    solution.set_scores(np.mean(final_y), feedback)
    end_time = time.time()
    print(f"evaluatePhotonic() time taken: {end_time - start_time} seconds")

    return solution


if __name__ == "__main__":
    ai_model = sys.argv[1]  # gpt-4-turbo or gpt-3.5-turbo gpt-4o llama3:70b
    parent_size = int(sys.argv[2])
    offspring_size = int(sys.argv[3])
    es_flag = False if sys.argv[4] == "0" else True
    if "deepseek" in ai_model:
        # api_key = os.getenv("DEEPSEEK_API_KEY")
        api_key = os.getenv("TENCENT_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    model = llm.OpenAI_LLM(api_key=api_key, model=ai_model)
    n3 = " + " if es_flag else ", "
    experiment_name = f"GNBG_({parent_size}{n3}{offspring_size})"

    description = ""
    algorithmic_insight = ""
    task_prompt = f"""
    The optimization algorithm should be able to find high-performing solutions to a wide range of tasks, which include evaluation on real-world applications such as, e.g., optimization of multilayered photonic structures. {description}{algorithmic_insight}Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
    The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between func.bounds.lb (lower bound) and func.bounds.ub (upper bound). The dimensionality can be varied.
    Give an excellent and novel heuristic algorithm to solve this task and include it's one-line description with the main idea of the algorithm.
    """
    print("Using model:", model)
    es = LLaMEA(evaluateGNBG, llm=model, n_parents=parent_size,
                n_offspring=offspring_size, task_prompt=task_prompt,
                experiment_name=experiment_name, elitism=es_flag,
                HPO=False, budget=100, minimization=True)
    print(es.run())
