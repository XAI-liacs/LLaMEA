import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.max_iter = budget // self.num_particles

    def __call__(self, func):
        def pso_optimize(obj_func):
            # PSO initialization
            # PSO optimization steps
            
        def sa_optimize(obj_func):
            # SA initialization
            # SA optimization steps
            
        best_solution = None
        best_fitness = np.inf

        for _ in range(self.max_iter):
            pso_solution, pso_fitness = pso_optimize(func)
            sa_solution, sa_fitness = sa_optimize(func)
            
            if pso_fitness < best_fitness:
                best_solution = pso_solution
                best_fitness = pso_fitness
            
            if sa_fitness < best_fitness:
                best_solution = sa_solution
                best_fitness = sa_fitness

        return best_solution