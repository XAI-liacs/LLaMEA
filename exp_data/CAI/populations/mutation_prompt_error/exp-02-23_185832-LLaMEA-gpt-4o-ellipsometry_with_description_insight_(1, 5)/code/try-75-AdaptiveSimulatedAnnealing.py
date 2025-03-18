import numpy as np
from scipy.optimize import minimize

class AdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Initial random solution
        current_solution = lb + (ub - lb) * np.random.rand(self.dim)
        current_value = func(current_solution)
        self.budget -= 1

        # Annealing parameters
        initial_temp = 1.0
        final_temp = 0.01
        temp_decay = (final_temp / initial_temp) ** (1.0 / (self.budget // 2))
        temperature = initial_temp

        def local_optimization(x0):
            result = minimize(func, x0, method='BFGS', bounds=list(zip(lb, ub)),
                              options={'maxiter': 10, 'gtol': 1e-6})
            return result.x, result.fun

        while self.budget > 0:
            # Generate a new candidate solution
            perturbation = np.random.normal(0, temperature, size=self.dim)
            candidate_solution = np.clip(current_solution + perturbation, lb, ub)
            candidate_value = func(candidate_solution)
            self.budget -= 1

            # Acceptance criterion
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / temperature):
                current_solution, current_value = candidate_solution, candidate_value

            # Local optimization with remaining budget
            if self.budget > 0 and np.random.rand() < 0.1:
                optimized_solution, optimized_value = local_optimization(current_solution)
                if optimized_value < current_value:
                    current_solution, current_value = optimized_solution, optimized_value
                self.budget -= 10

            # Anneal temperature
            temperature *= temp_decay

        return current_solution, current_value