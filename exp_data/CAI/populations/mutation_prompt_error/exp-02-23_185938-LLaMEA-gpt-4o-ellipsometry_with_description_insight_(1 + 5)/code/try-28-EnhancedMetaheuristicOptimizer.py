import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_sample_count = max(10, self.budget // 6)  # Adjust initial sample count for more exploration
        
        # Randomly sample initial points within bounds
        initial_samples = []
        for _ in range(initial_sample_count):
            sample = np.array([
                np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
            ])
            initial_samples.append(sample)

        # Evaluate initial samples and find the best one
        best_sample = None
        best_value = float('inf')
        for sample in initial_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        # Introduce additional stochastic perturbations around the best initial sample
        perturbed_samples = [best_sample + np.random.normal(0, 0.05, self.dim) for _ in range(3)]
        for sample in perturbed_samples:
            if self.budget <= 0:
                break
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample

        # Narrow bounds around the best sample with more aggressive adjustment
        bounds = [(max(lb, x - 0.2 * (ub - lb)), min(ub, x + 0.2 * (ub - lb))) 
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use L-BFGS-B for local optimization with early stopping
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.7), 'ftol': 1e-7})

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails