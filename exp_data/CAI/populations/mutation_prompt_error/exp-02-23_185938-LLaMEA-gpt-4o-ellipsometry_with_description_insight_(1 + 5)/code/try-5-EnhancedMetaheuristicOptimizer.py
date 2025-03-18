import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize an adaptive sample size
        initial_sample_count = 10
        
        # Function to dynamically adjust the sample size
        def dynamic_sample_size(remaining_budget):
            return max(10, remaining_budget // 10)

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

        # Dynamically expand initial exploration based on variance reduction
        while self.budget > self.budget // 2:
            new_sample_count = dynamic_sample_size(self.budget)
            for _ in range(new_sample_count):
                sample = np.array([
                    np.random.uniform(lb, ub) for lb, ub in zip(func.bounds.lb, func.bounds.ub)
                ])
                value = func(sample)
                self.budget -= 1
                if value < best_value:
                    best_value = value
                    best_sample = sample
                if self.budget <= 0:
                    return best_sample

        # Memory-based approach to refine bounds
        history = [best_sample]
        bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for local optimizers
        def objective(x):
            return func(x)

        # Alternate between Nelder-Mead and BFGS for local optimization
        for _ in range(2):
            res = minimize(objective, x0=best_sample, method='Nelder-Mead', options={'maxfev': self.budget // 2})
            best_sample = res.x
            history.append(best_sample)
            self.budget -= res.nfev

            if self.budget <= 0:
                return best_sample

            res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget})
            best_sample = res.x
            history.append(best_sample)
            self.budget -= res.nfev

            if self.budget <= 0:
                return best_sample

            # Adaptive bounds refinement based on memory
            bounds = [(max(lb, x - 0.1 * (ub - lb)), min(ub, x + 0.1 * (ub - lb)))
                      for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        return best_sample