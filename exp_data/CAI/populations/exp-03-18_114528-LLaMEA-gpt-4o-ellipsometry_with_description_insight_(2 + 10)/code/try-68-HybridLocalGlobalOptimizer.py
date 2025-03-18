import numpy as np
from scipy.optimize import minimize

class HybridLocalGlobalOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Enhanced initial sampling using differential evolution strategy
        num_initial_samples = min(20, self.budget // 4)
        initial_population = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        for i in range(num_initial_samples // 2):
            idx1, idx2, idx3 = np.random.choice(num_initial_samples, 3, replace=False)
            mutant = initial_population[idx1] + 0.8 * (initial_population[idx2] - initial_population[idx3])
            mutant = np.clip(mutant, lb, ub)
            if func(mutant) < func(initial_population[idx1]):
                initial_population[idx1] = mutant

        best_solution = None
        best_value = float('inf')
        evaluations = 0

        for point in initial_population:
            # Local optimization using BFGS with dynamic constraint adjustment
            res = minimize(func, point, method='BFGS',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': max(3, (self.budget - evaluations) // num_initial_samples)})
            evaluations += res.nfev

            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

            if evaluations >= self.budget:
                break

        return best_solution