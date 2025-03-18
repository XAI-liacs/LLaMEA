import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def differential_evolution(self, func, bounds, pop_size=10, mut=0.8, crossp=0.7):
        pop = np.random.rand(pop_size, self.dim)  # Initialize population
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([func(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        
        for i in range(self.budget - pop_size):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
                cross_points = np.random.rand(self.dim) < crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                f = func(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
        return best

    def local_optimize(self, func, x0, bounds):
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        return result.x, result.fun

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        # Stage 1: Global exploration using Differential Evolution
        best_global = self.differential_evolution(func, bounds)
        
        # Stage 2: Local exploitation with BFGS
        best_local, _ = self.local_optimize(func, best_global, bounds)
        
        return best_local