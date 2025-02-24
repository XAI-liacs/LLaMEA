import numpy as np
from scipy.optimize import minimize

class IncrementalHybridSamplingLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial uniform random sampling with a small number of samples
        num_initial_samples = 5
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_initial_samples, self.dim))
        evaluations = []

        # Evaluate initial samples
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        # Sort initial samples based on their function value
        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        # Define a local optimization function using Nelder-Mead
        def local_optimization(x0):
            res = minimize(func, x0, method='Nelder-Mead', options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        # Conduct local optimization from the best initial sample
        if len(evaluations) < self.budget:
            solution, value = local_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        # Iterative sampling and local optimization
        while len(evaluations) < self.budget:
            new_samples_count = min(self.budget // 10, len(evaluations) + 5)
            new_samples = np.random.uniform(lower_bounds, upper_bounds, size=(new_samples_count, self.dim))

            for sample in new_samples:
                if len(evaluations) < self.budget:
                    evaluations.append((sample, func(sample)))
                else:
                    break

            # Sort new samples and update best sample
            evaluations.sort(key=lambda x: x[1])
            current_best_sample, current_best_value = evaluations[0]

            if current_best_value < best_value:
                best_sample, best_value = current_best_sample, current_best_value

            # Additional local optimization
            solution, value = local_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        return best_sample, best_value