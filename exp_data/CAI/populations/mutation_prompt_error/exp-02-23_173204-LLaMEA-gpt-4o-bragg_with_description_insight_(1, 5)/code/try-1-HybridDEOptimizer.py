import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for DE
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.best_solution = None
        self.best_score = float('inf')

    def quasi_oppositional_initialization(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - population
        combined_population = np.vstack((population, opposite_population))
        return combined_population

    def periodicity_enforcement(self, solution):
        segment_size = max(2, self.dim // 5)  # Dynamically chosen segment size
        return np.repeat(np.mean(solution.reshape(-1, segment_size), axis=1), segment_size)

    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = self.quasi_oppositional_initialization(lb, ub)

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                # Periodicity enforcement
                crossover = self.periodicity_enforcement(crossover)

                # Selection
                score = func(crossover)
                if score < func(population[i]):
                    population[i] = crossover
                    if score < self.best_score:
                        self.best_score = score
                        self.best_solution = crossover

    def local_optimization(self, func, initial_solution, bounds):
        result = minimize(func, initial_solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        if result.fun < self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        bounds = func.bounds
        self.differential_evolution(func, bounds)
        
        # Perform local optimization on the best found solution
        if self.best_solution is not None:
            self.local_optimization(func, self.best_solution, bounds)
        
        return self.best_solution