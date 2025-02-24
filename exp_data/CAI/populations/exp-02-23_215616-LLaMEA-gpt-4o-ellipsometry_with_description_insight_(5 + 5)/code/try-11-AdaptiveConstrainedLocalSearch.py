import numpy as np
from scipy.optimize import minimize

class AdaptiveConstrainedLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds for the search space
        lower_bounds = np.array(func.bounds.lb)
        upper_bounds = np.array(func.bounds.ub)

        # Initial uniform random sampling with more samples
        num_initial_samples = min(self.budget // 4, 12)  # Change: Dynamic sampling count for better exploration
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

        # Define a local optimization function using BFGS
        def local_optimization(x0):
            res = minimize(func, x0, method='L-BFGS-B', bounds=list(zip(lower_bounds, upper_bounds)),
                           options={'maxiter': self.budget - len(evaluations)})
            return res.x, res.fun

        # Conduct local optimization from the best initial sample
        if len(evaluations) < self.budget:
            solution, value = local_optimization(best_sample)
            if value < best_value:
                best_sample, best_value = solution, value

        # Adjust bounds based on best found solution
        margin = 0.05 + 0.05 * np.random.rand()  # Keep this line unchanged
        new_lower_bounds = np.maximum(lower_bounds, best_sample - margin * (best_sample - lower_bounds))  # Change: Adaptive bounds adjustment
        new_upper_bounds = np.minimum(upper_bounds, best_sample + margin * (upper_bounds - best_sample))  # Change: Adaptive bounds adjustment

        # Final local optimization with adjusted bounds
        if len(evaluations) < self.budget:
            final_solution, final_value = local_optimization(best_sample)
            if final_value < best_value:
                best_sample, best_value = final_solution, final_value

        return best_sample, best_value