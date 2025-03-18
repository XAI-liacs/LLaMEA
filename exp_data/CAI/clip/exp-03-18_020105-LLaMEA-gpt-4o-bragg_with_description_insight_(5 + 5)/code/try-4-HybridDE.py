import numpy as np
from scipy.optimize import minimize

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size for DE
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability

    def _de_step(self, population, func):
        new_population = np.empty_like(population)
        for i in range(self.pop_size):
            x = population[i]
            a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(self.dim)] = True
            trial = np.where(cross_points, mutant, x)
            if func(trial) < func(x):
                new_population[i] = trial
            else:
                new_population[i] = x
        return new_population

    def _fine_tune(self, x, func):
        result = minimize(func, x, method='L-BFGS-B', bounds=list(zip(func.bounds.lb, func.bounds.ub)))
        return result.x if result.success else x

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Symmetric initialization
        population = lb + (ub - lb) * np.random.rand(self.pop_size, dim)
        quasi_opposite_population = ub + lb - population
        population = np.vstack((population, quasi_opposite_population))
        self.pop_size *= 2

        eval_count = 0
        while eval_count < self.budget:
            if eval_count + self.pop_size > self.budget:
                self.pop_size = self.budget - eval_count
            population = self._de_step(population, func)
            eval_count += self.pop_size
            if eval_count + self.pop_size > self.budget:
                break

        # Fine-tuning with BFGS
        best_solution = min(population, key=func)
        best_solution = self._fine_tune(best_solution, func)

        return best_solution