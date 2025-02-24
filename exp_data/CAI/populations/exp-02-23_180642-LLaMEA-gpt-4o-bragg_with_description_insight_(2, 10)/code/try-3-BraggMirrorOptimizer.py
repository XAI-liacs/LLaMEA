import numpy as np
from scipy.optimize import minimize

class BraggMirrorOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9):
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        scores = np.array([func(ind) for ind in pop])
        self.evaluations += pop_size

        while self.evaluations < self.budget:
            for i in range(pop_size):
                if self.evaluations >= self.budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < scores[i]:
                    pop[i], scores[i] = trial, trial_score

        best_idx = np.argmin(scores)
        return pop[best_idx], scores[best_idx]

    def local_search(self, func, x0, bounds):
        result = minimize(func, x0=x0, bounds=[(l, u) for l, u in zip(bounds.lb, bounds.ub)],
                          method='L-BFGS-B', options={'maxiter': self.budget - self.evaluations})
        self.evaluations += result.nfev
        return result.x, result.fun

    def adaptive_periodicity(self, solution):
        period = self.dim // 2
        if period > 0:
            # Correct the reshape size to ensure compatibility
            averaged_solution = np.tile(np.mean(solution[:period].reshape(-1, self.dim // period), axis=0), period)
            return averaged_solution[:self.dim]  # Ensure the solution length matches self.dim
        return solution

    def __call__(self, func):
        bounds = func.bounds
        pop_size = 20
        best_solution, best_score = self.differential_evolution(func, bounds, pop_size)
        periodic_solution = self.adaptive_periodicity(best_solution)
        final_solution, final_score = self.local_search(func, periodic_solution, bounds)
        return final_solution