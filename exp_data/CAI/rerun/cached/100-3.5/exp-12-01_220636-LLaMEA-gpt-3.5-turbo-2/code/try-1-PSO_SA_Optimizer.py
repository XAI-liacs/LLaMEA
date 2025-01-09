import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def pso_search():
            # PSO initialization
            ...

        def sa_search():
            # SA initialization
            ...

        best_solution = None
        best_fitness = np.inf

        for _ in range(self.max_iter):
            # PSO phase
            pso_solution, pso_fitness = pso_search()

            if pso_fitness < best_fitness:
                best_solution = pso_solution
                best_fitness = pso_fitness

            # SA phase
            sa_solution, sa_fitness = sa_search()

            if sa_fitness < best_fitness:
                best_solution = sa_solution
                best_fitness = sa_fitness

            if func.evaluations >= self.budget:
                break

        return best_solution