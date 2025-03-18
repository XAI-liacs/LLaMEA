import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def periodicity_aware_init(self, lb, ub):
        # Initialize population with periodic patterns
        base_pattern = np.linspace(lb, ub, self.dim // 2)
        population = np.tile(base_pattern, (self.population_size, 2))
        return population + np.random.uniform(-0.01, 0.01, (self.population_size, self.dim))

    def differential_evolution(self, func, lb, ub):
        population = self.periodicity_aware_init(lb, ub)
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget - len(population)):
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def local_refinement(self, x, func, lb, ub):
        res = minimize(func, x, method='L-BFGS-B', bounds=[(lb, ub)] * self.dim)
        return res.x if res.success else x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = self.differential_evolution(func, lb, ub)
        best_solution = self.local_refinement(best_solution, func, lb, ub)
        return best_solution