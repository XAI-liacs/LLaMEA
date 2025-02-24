import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.best_solution = None
        self.best_score = float('inf')
        self.elite_archive = None

    def quasi_oppositional_initialization(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - population
        combined_population = np.vstack((population, opposite_population))
        return combined_population

    def periodicity_enforcement(self, solution, period=2):
        return np.tile(np.mean(solution.reshape(-1, period), axis=1), period)

    def adaptive_parameters(self, generation):
        cycle_length = self.budget // (3 * self.population_size)
        phase = (generation % cycle_length) / cycle_length
        self.F = 0.7 + 0.3 * np.abs(np.sin(2 * np.pi * phase))  # Enhanced F
        self.CR = 0.8 + 0.2 * np.cos(2 * np.pi * phase)  # Enhanced CR

    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        population = self.quasi_oppositional_initialization(lb, ub)

        for generation in range(self.budget // self.population_size):
            self.adaptive_parameters(generation)

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                crossover = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])

                if generation % 5 == 0 or generation % 7 == 0:  # Enhanced periodicity condition
                    crossover = self.periodicity_enforcement(crossover)

                score = func(crossover)
                if score < func(population[i]):
                    population[i] = crossover
                    if score < self.best_score:
                        self.best_score = score
                        self.best_solution = crossover
                        self.elite_archive = crossover.copy()
                else:
                    population[i] = np.random.uniform(lb, ub, self.dim)

            if generation % 4 == 0 and self.best_solution is not None:
                self.local_optimization(func, self.best_solution, bounds)

    def local_optimization(self, func, initial_solution, bounds):
        result = minimize(func, initial_solution, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        if result.fun < self.best_score:
            self.best_score = result.fun
            self.best_solution = result.x

    def __call__(self, func):
        bounds = func.bounds
        self.differential_evolution(func, bounds)
        
        if self.best_solution is not None:
            self.local_optimization(func, self.best_solution, bounds)
        
        return self.best_solution