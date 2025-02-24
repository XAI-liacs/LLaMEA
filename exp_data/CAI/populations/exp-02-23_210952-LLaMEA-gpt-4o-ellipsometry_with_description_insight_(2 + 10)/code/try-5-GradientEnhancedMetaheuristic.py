import numpy as np
from scipy.optimize import minimize

class GradientEnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract the bounds and prepare for optimizations
        lower_bounds = func.bounds.lb
        upper_bounds = func.bounds.ub
        bounds = [(low, high) for low, high in zip(lower_bounds, upper_bounds)]
        
        # Calculate the number of initial samples based on the available budget
        num_initial_samples = min(self.budget // 3, 10)
        remaining_budget = self.budget - num_initial_samples

        # Initialize the best solution found so far
        best_solution = None
        best_score = float('inf')

        # Step 1: Uniformly sample the initial solutions
        initial_solutions = np.random.uniform(lower_bounds, upper_bounds, (num_initial_samples, self.dim))
        
        for solution in initial_solutions:
            score = func(solution)
            if score < best_score:
                best_score = score
                best_solution = solution
        
        # Step 2: Use finite difference to estimate gradient
        def gradient_estimation(x):
            epsilon = 1e-8
            grad = np.zeros_like(x)
            for i in range(self.dim):
                x_plus = np.array(x, copy=True)
                x_plus[i] += epsilon
                grad[i] = (func(x_plus) - func(x)) / epsilon
            return grad

        # Step 3: Use BFGS with gradient from best initial sample
        def wrapped_func(x):
            nonlocal remaining_budget
            if remaining_budget <= 0:
                return float('inf')
            remaining_budget -= 1
            return func(x)

        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', jac=gradient_estimation, bounds=bounds, options={'maxfun': remaining_budget})

        # Step 4: Adjust bounds for further exploration if budget allows
        if remaining_budget > 0:
            adjust_factor = 0.1
            adjusted_bounds = [(max(low, x - adjust_factor * (high - low)), min(high, x + adjust_factor * (high - low))) for x, (low, high) in zip(result.x, bounds)]
            result = minimize(wrapped_func, result.x, method='L-BFGS-B', jac=gradient_estimation, bounds=adjusted_bounds, options={'maxfun': remaining_budget})

        # Return the best found solution
        return result.x if result.success else best_solution