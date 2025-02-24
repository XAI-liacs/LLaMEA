import numpy as np
from scipy.optimize import minimize

class RefinedBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        func_calls = 0
        # Extract bounds
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Define sampling fractions for exploration
        fraction_for_sampling = 0.2
        num_samples = int(self.budget * fraction_for_sampling)

        # Latin Hypercube Sampling for better initial coverage
        samples = self.latin_hypercube_sampling(lb, ub, num_samples)
        sample_evaluations = [func(sample) for sample in samples]
        func_calls += num_samples

        # Select the best sample as the starting point for local optimization
        best_sample_index = np.argmin(sample_evaluations)
        best_initial_guess = samples[best_sample_index]

        # Define the objective function to optimize
        def objective(x):
            nonlocal func_calls
            if func_calls >= self.budget:
                return float('inf')
            func_calls += 1
            return func(x)

        # Perform local optimization using L-BFGS-B method
        result = minimize(objective, best_initial_guess, method='L-BFGS-B', bounds=[(l, u) for l, u in zip(lb, ub)])

        # Return best solution found
        return result.x if result.success else best_initial_guess

    def latin_hypercube_sampling(self, lb, ub, num_samples):
        intervals = np.linspace(0, 1, num_samples + 1)
        lower_intervals = intervals[:-1]
        upper_intervals = intervals[1:]
        points = np.random.uniform(lower_intervals, upper_intervals, (num_samples, self.dim))
        np.random.shuffle(points)
        return lb + (ub - lb) * points