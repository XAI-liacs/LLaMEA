import numpy as np
from scipy.optimize import minimize

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.bounds = None
        self.evals = 0

    def _initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.bounds = (lb, ub)

    def _periodicity_penalty(self, x):
        # Encourage periodic solutions by penalizing deviations from periodic pattern
        half_dim = self.dim // 2
        periodic_pattern = np.tile(x[:half_dim], 2)
        return np.sum((x - periodic_pattern) ** 2)

    def _mutate(self, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_optimize(self, x, func):
        result = minimize(func, x, method='L-BFGS-B', bounds=[(lb, ub) for lb, ub in zip(self.bounds[0], self.bounds[1])])
        return result.x

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(lb, ub)

        best_solution = None
        best_score = float('inf')

        while self.evals < self.budget:
            for i in range(self.pop_size):
                target = self.population[i]

                mutant = self._mutate(i)
                trial = self._crossover(target, mutant)

                penalty = self._periodicity_penalty(trial)
                trial_score = func(trial) + penalty
                self.evals += 1

                if trial_score < best_score:
                    best_solution = trial
                    best_score = trial_score

                if trial_score < func(target) + self._periodicity_penalty(target):
                    self.population[i] = trial

                if self.evals >= self.budget:
                    break

            if self.evals < self.budget:
                refined_solution = self._local_optimize(best_solution, lambda x: func(x) + self._periodicity_penalty(x))
                refined_score = func(refined_solution) + self._periodicity_penalty(refined_solution)
                self.evals += 1

                if refined_score < best_score:
                    best_solution = refined_solution
                    best_score = refined_score

        return best_solution