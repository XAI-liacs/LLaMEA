import numpy as np
from scipy.optimize import minimize

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples with hybrid approach
        initial_sample_count = max(15, self.budget // 7)  # Adjusted sample count
        
        # Randomly sample initial points within bounds (hybrid strategy)
        initial_samples = [
            np.random.uniform(func.bounds.lb, func.bounds.ub) for _ in range(initial_sample_count // 2)
        ]
        halton_samples = self.halton_sequence(initial_sample_count // 2, func.bounds.lb, func.bounds.ub)
        initial_samples.extend(halton_samples)

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

        # Dynamic bounds adjustment strategy
        bounds = [(max(lb, x - 0.10 * (ub - lb)), min(ub, x + 0.10 * (ub - lb)))
                  for x, lb, ub in zip(best_sample, func.bounds.lb, func.bounds.ub)]

        # Define the objective function for the local optimizer
        def objective(x):
            return func(x)

        # Use L-BFGS-B for local optimization with adaptive options
        res = minimize(objective, x0=best_sample, method='L-BFGS-B', bounds=bounds, options={'maxfun': int(self.budget * 0.85), 'ftol': 1e-8})  # Fine-tuned ftol

        if res.success:
            return res.x
        else:
            return best_sample  # Fallback if optimization fails

    def halton_sequence(self, count, lb, ub):
        """Generate Halton sequence samples within given bounds."""
        # Implementation of a basic Halton sequence generator for 2D
        samples = []
        for i in range(count):
            base2 = self.van_der_corput(i, 2)
            base3 = self.van_der_corput(i, 3)
            sample = [(ub[j] - lb[j]) * base + lb[j] for j, base in enumerate([base2, base3])]
            samples.append(sample)
        return samples

    def van_der_corput(self, n, base):
        """Van der Corput sequence implementation."""
        vdc = 0
        denom = 1
        while n > 0:
            denom *= base
            n, remainder = divmod(n, base)
            vdc += remainder / denom
        return vdc