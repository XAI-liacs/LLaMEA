import numpy as np
from scipy.optimize import minimize

class AdaptiveBoundaryMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples
        initial_sample_count = max(10, self.budget // 10)
        
        # Randomly sample initial points within bounds
        initial_samples = np.array([
            [np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)]
            for _ in range(initial_sample_count)
        ])

        # Evaluate initial samples and find the best one
        sample_values = np.array([func(sample) for sample in initial_samples])
        self.budget -= initial_sample_count
        best_value_index = np.argmin(sample_values)
        best_sample = initial_samples[best_value_index]

        if self.budget <= 0:
            return best_sample

        # Calculate sample spread to adaptively scale bounds
        sample_spread = np.std(initial_samples, axis=0)
        adaptive_scaling = 0.1 + 0.5 * sample_spread / (np.array(func.bounds.ub) - np.array(func.bounds.lb))
        
        # Narrow bounds adaptively around the best initial sample
        bounds = [(max(lb, best_sample[i] - adaptive_scaling[i] * (ub - lb)), 
                   min(ub, best_sample[i] + adaptive_scaling[i] * (ub - lb)))
                  for i, (lb, ub) in enumerate(zip(func.bounds.lb, func.bounds.ub))]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use BFGS for local optimization within the new adaptive bounds
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget, 'ftol': 1e-6})

        return res.x