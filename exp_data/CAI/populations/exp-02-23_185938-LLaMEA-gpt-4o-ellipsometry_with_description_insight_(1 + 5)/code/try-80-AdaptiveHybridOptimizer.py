import numpy as np
from scipy.optimize import minimize

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate initial sample count using dimensionality
        initial_sample_count = max(5, self.budget // 10)

        # Randomly sample initial points within bounds
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        # Evaluate initial samples and select the top performers
        evaluated_samples = []
        for sample in initial_samples:
            value = func(sample)
            self.budget -= 1
            evaluated_samples.append((value, sample))
            if self.budget <= 0:
                break

        # Sort by performance and select top samples
        evaluated_samples.sort()
        top_samples = [s for _, s in evaluated_samples[:min(3, len(evaluated_samples))]]

        # Define a local search using multiple top samples
        def local_search(x0):
            bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                      for x, lb, ub in zip(x0, func.bounds.lb, func.bounds.ub)]
            res = minimize(func, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.7 / len(top_samples)), 'ftol': 1e-7})
            return res

        best_value = float('inf')
        best_solution = None
        for sample in top_samples:
            if self.budget <= 0:
                break
            res = local_search(sample)
            if res.success and res.fun < best_value:
                best_value = res.fun
                best_solution = res.x

        return best_solution if best_solution is not None else top_samples[0]