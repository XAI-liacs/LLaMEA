import numpy as np
from scipy.optimize import minimize

class MultiGradientSamplingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial sampling size based on budget
        num_initial_samples = min(self.budget // 3, 20)
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

        def multi_gradient_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        # Conduct local optimization from the best initial samples
        for i in range(min(3, len(evaluations))):  # Use top 3 samples
            if len(evaluations) < self.budget:
                solution, value = multi_gradient_optimization(evaluations[i][0])
                if value < best_value:
                    best_sample, best_value = solution, value

        # Adjust bounds based on the best-found solution
        margin_factor = 0.1
        new_lower_bounds = np.maximum(lower_bounds, best_sample - margin_factor * best_sample)
        new_upper_bounds = np.minimum(upper_bounds, best_sample + margin_factor * best_sample)

        # Final local optimization with adjusted bounds
        if len(evaluations) < self.budget:
            final_solution, final_value = multi_gradient_optimization(best_sample)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        return best_sample, best_value