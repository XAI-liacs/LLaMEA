import numpy as np
from scipy.optimize import minimize

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def _initialize_population(self, bounds):
        lb, ub = bounds
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def _mutate(self, target_idx, population):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _enforce_periodicity(self, solution):
        period = self.dim // 2
        solution[:period] = solution[period:]
        return solution

    def _local_search(self, func, solution):
        res = minimize(func, solution, method='BFGS', bounds=func.bounds)
        return res.x if res.success else solution

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = self._initialize_population(bounds)
        best_idx = np.argmin([func(indiv) for indiv in population])
        best = population[best_idx]
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                target = population[i]
                mutant = self._mutate(i, population)
                trial = self._crossover(target, mutant)
                trial = np.clip(trial, *bounds)  # Ensure within bounds
                trial = self._enforce_periodicity(trial)
                f_trial = func(trial)
                evaluations += 1
                if f_trial < func(target):
                    population[i] = trial
                    if f_trial < func(best):
                        best = trial
            # Perform a local search on the best current solution
            best = self._local_search(func, best)
            evaluations += 1
        
        return best