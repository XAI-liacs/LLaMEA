import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for DE
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.used_budget = 0

    def _quasi_oppositional_init(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        midpoint = (lb + ub) / 2.0
        q_opposite_population = midpoint + (midpoint - population)
        return np.vstack((population, q_opposite_population))

    def _de_step(self, population, func, bounds):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self._evaluate(trial, func) < self._evaluate(population[i], func):
                new_population[i] = trial
            else:
                new_population[i] = population[i]
        return new_population

    def _local_search(self, best_candidate, func, bounds):
        result = minimize(func, best_candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
        return result.x if result.success else best_candidate

    def _evaluate(self, candidate, func):
        if self.used_budget >= self.budget:
            return float('inf')
        self.used_budget += 1
        return func(candidate)

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        population = self._quasi_oppositional_init(lb, ub)

        while self.used_budget < self.budget:
            population = self._de_step(population, func, bounds)
            best_candidate = min(population, key=lambda x: self._evaluate(x, func))
            best_candidate = self._local_search(best_candidate, func, bounds)
            if self.used_budget >= self.budget:
                break

        best_solution = min(population, key=lambda x: self._evaluate(x, func))
        return best_solution