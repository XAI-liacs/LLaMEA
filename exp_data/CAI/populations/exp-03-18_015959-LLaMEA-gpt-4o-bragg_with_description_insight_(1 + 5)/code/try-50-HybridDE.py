import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.4, 0.9
        self.used_budget = 0

    def _quasi_oppositional_init(self, lb, ub):
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        midpoint = (lb + ub) / 2.0
        q_opposite_population = midpoint + (midpoint - population)
        return np.vstack((population, q_opposite_population))

    def _adaptive_parameters(self):
        adapt_ratio = self.used_budget / self.budget
        F = self.F_min + adapt_ratio * (self.F_max - self.F_min)
        CR = self.CR_max - adapt_ratio * (self.CR_max - self.CR_min)
        return F, CR

    def _dynamic_population(self, population):
        # Reducing population size dynamically as budget depletes
        if self.used_budget > 0.5 * self.budget:
            return population[:self.population_size // 2]
        return population

    def _de_step(self, population, func, bounds):
        new_population = np.empty_like(population)
        F, CR = self._adaptive_parameters()
        for i in range(len(population)):
            indices = [idx for idx in range(len(population)) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if self._evaluate(trial, func) < self._evaluate(population[i], func):
                new_population[i] = trial
            else:
                new_population[i] = population[i]
        return new_population

    def _local_search(self, best_candidate, func, bounds):
        if np.random.rand() < 0.5:
            result = minimize(func, best_candidate, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B')
            if result.success:
                return result.x
        # Stronger periodic enforcement with slight randomization
        period_len = self.dim // 10
        periodic_candidate = np.tile(best_candidate[:period_len], self.dim // period_len)
        periodic_candidate += 0.01 * np.random.randn(self.dim)  # Random noise
        return periodic_candidate if self._evaluate(periodic_candidate, func) < self._evaluate(best_candidate, func) else best_candidate

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
            population = self._dynamic_population(population)  # Apply dynamic population resizing
            best_candidate = min(population, key=lambda x: self._evaluate(x, func))
            best_candidate = self._local_search(best_candidate, func, bounds)
            if self.used_budget >= self.budget:
                break

        best_solution = min(population, key=lambda x: self._evaluate(x, func))
        return best_solution