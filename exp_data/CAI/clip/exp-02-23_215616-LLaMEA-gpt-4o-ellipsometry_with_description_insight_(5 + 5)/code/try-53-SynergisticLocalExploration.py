import numpy as np
from scipy.optimize import minimize

class SynergisticLocalExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial uniform random sampling with adaptive sample size
        num_initial_samples = min(max(self.budget // 10, 5), 20)  # Adaptive initial sample number
        samples = np.random.uniform(lower_bounds, upper_bounds, size=(num_initial_samples, self.dim))

        # Evaluate the initial samples
        evaluations = []
        for sample in samples:
            if len(evaluations) < self.budget:
                evaluations.append((sample, func(sample)))
            else:
                break

        # Sort initial samples based on their function value
        evaluations.sort(key=lambda x: x[1])
        best_sample, best_value = evaluations[0]

        # Dynamic descent rate based on remaining budget
        descent_scale = max(0.1, (self.budget - len(evaluations)) / self.budget)

        # Define a local optimization function with adaptive descent rate
        def local_optimization(x0):
            options = {'maxiter': int((self.budget - len(evaluations)) * descent_scale)}
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)), options=options)
            return res.x, res.fun

        # Conduct progressive local optimizations
        while len(evaluations) < self.budget:
            solution, value = local_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        return best_sample, best_value