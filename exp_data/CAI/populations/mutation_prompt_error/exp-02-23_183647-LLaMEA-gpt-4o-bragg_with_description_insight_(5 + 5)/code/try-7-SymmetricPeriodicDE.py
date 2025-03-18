import numpy as np
from scipy.optimize import minimize

class SymmetricPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Number of candidate solutions
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = None
        self.func_evaluations = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        center = (lb + ub) / 2
        range_half = (ub - lb) / 2
        # Symmetric initialization: around the mid-point of the search space
        self.population = center + np.random.uniform(-range_half, range_half, (self.population_size, self.dim))

    def mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, 0, 1)  # Assuming normalized [0, 1] bounds for mutation

    def crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def local_optimization(self, candidate, func):
        # Encourage periodicity by optimizing a cost that penalizes deviation from periodic patterns
        def periodic_cost(x):
            period = self.dim // 2  # Example: Half the dimension as a period
            deviation = np.sum((x[:period] - x[period:]) ** 2)
            return func(x) + deviation

        result = minimize(periodic_cost, candidate, bounds=func.bounds)
        return result.x

    def __call__(self, func):
        self.initialize_population(func.bounds)
        best_solution = None
        best_score = float('inf')

        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                if self.func_evaluations >= self.budget:
                    break
                
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)

                trial = self.local_optimization(trial, func)

                trial_score = func(trial)
                self.func_evaluations += 1

                if trial_score < best_score:
                    best_score = trial_score
                    best_solution = trial

                # Selection
                if trial_score < func(target):
                    self.population[i] = trial

        return best_solution