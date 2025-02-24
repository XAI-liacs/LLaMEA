import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        func_calls = 0
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Dynamic adjustment of sampling fraction based on budget usage
        fraction_for_sampling = max(0.2, 1 - (func_calls / self.budget))
        num_samples = int(self.budget * fraction_for_sampling)

        # Uniform random sampling to initialize search
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        sample_evaluations = [func(sample) for sample in samples]
        func_calls += num_samples

        # Select the best sample as the starting point for local optimization
        best_sample_index = np.argmin(sample_evaluations)
        best_initial_guess = samples[best_sample_index]

        # Define the function to optimize using initial guess
        def objective(x):
            nonlocal func_calls
            if func_calls >= self.budget:
                return float('inf')
            func_calls += 1
            return func(x)

        # Perform local optimization using Nelder-Mead method
        result = minimize(objective, best_initial_guess, method='Nelder-Mead', bounds=[(l, u) for l, u in zip(lb, ub)])

        # Return best solution found
        return result.x if result.success else best_initial_guess