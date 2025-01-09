import numpy as np

class AntBeeOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.rho = 0.1
        self.num_ants = 10
        self.num_bees = 10

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            # Ant phase
            ant_solutions = [best_solution + np.random.uniform(-self.rho, self.rho, self.dim) for _ in range(self.num_ants)]
            ant_solutions = [np.clip(solution, self.lower_bound, self.upper_bound) for solution in ant_solutions]
            ant_fitness = [func(solution) for solution in ant_solutions]
            ant_best_idx = np.argmin(ant_fitness)
            
            # Bee phase
            bee_solutions = [best_solution + np.random.normal(0, 1, self.dim) for _ in range(self.num_bees)]
            bee_solutions = [np.clip(solution, self.lower_bound, self.upper_bound) for solution in bee_solutions]
            bee_fitness = [func(solution) for solution in bee_solutions]
            bee_best_idx = np.argmin(bee_fitness)
            
            if ant_fitness[ant_best_idx] < best_fitness:
                best_solution = ant_solutions[ant_best_idx]
                best_fitness = ant_fitness[ant_best_idx]
            if bee_fitness[bee_best_idx] < best_fitness:
                best_solution = bee_solutions[bee_best_idx]
                best_fitness = bee_fitness[bee_best_idx]
        
        return best_solution