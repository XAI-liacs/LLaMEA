import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def differential_evolution(self, func, bounds, pop_size=20, F=0.8, CR=0.9):
        pop = np.random.uniform(bounds.lb, bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        elite = best.copy()

        while self.eval_count < self.budget:
            F = 0.3 + 0.4 * (self.budget - self.eval_count) / self.budget
            CR = 0.5 + np.std(fitness) / np.mean(fitness)  # Change 1: Adaptive CR based on diversity
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial
                        if trial_fitness < func(elite):
                            elite = trial.copy()
            pop_size = max(10, int(pop_size * (self.budget - self.eval_count) / self.budget))
        return elite

    def local_search(self, func, x0, bounds):
        starts = [x0]  # Change 2: Multi-start with initial best
        starts += [np.random.uniform(bounds.lb, bounds.ub, self.dim) for _ in range(2)]  # Change 3: Additional starts
        best_result = None
        for start in starts:
            result = minimize(func, start, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                              method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-5})
            if best_result is None or result.fun < best_result.fun:
                best_result = result  # Change 4: Track best result across starts
        return best_result.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        final_solution = self.local_search(func, best_solution, bounds)
        return final_solution