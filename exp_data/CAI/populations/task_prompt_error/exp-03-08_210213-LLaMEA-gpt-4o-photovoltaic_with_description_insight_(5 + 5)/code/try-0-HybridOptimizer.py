import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = None

    def _differential_evolution(self, func, bounds, pop_size=15, F=0.8, CR=0.7):
        population = np.random.rand(pop_size, self.dim)
        for i in range(self.dim):
            population[:, i] = bounds.lb[i] + population[:, i] * (bounds.ub[i] - bounds.lb[i])

        scores = np.array([func(ind) for ind in population])
        best_idx = np.argmin(scores)
        best = population[best_idx]
        
        evaluations = pop_size

        while evaluations < self.budget:
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                score = func(trial)
                evaluations += 1
                if score < scores[i]:
                    scores[i] = score
                    population[i] = trial
                    if score < scores[best_idx]:
                        best_idx = i
                        best = trial
                if evaluations >= self.budget:
                    break
                    
        return best, scores[best_idx]

    def _local_search(self, func, x0):
        result = minimize(func, x0, bounds=self.bounds)
        return result.x, result.fun

    def __call__(self, func):
        self.bounds = func.bounds
        best_solution, best_score = self._differential_evolution(func, func.bounds)
        best_solution, best_score = self._local_search(func, best_solution)
        return best_solution