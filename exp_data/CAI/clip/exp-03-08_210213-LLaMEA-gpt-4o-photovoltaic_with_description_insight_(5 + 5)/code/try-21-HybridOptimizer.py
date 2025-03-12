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
        
        while self.eval_count < self.budget:
            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break
                # Change 1: Dynamically adjust F based on progress
                F = 0.9 - 0.5 * (self.eval_count / self.budget)
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
        return best

    def local_search(self, func, x0, bounds):
        # Change 2: Switching from SLSQP to L-BFGS-B with warm-start
        result = minimize(func, x0, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)],
                          method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-3})
        return result.x

    def __call__(self, func):
        bounds = func.bounds
        best_solution = self.differential_evolution(func, bounds)
        final_solution = self.local_search(func, best_solution, bounds)
        return final_solution