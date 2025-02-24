import numpy as np
from scipy.optimize import minimize

class AdaptiveGradientDescentWithBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)
        
        best_solution = initial_guess
        best_value = func(initial_guess)
        evaluations = 1  # Start with 1 evaluation for the initial guess

        # Perform optimization with adaptive boundary refinement
        while evaluations < self.budget:
            # Gradient Descent Step
            result = minimize(
                func, 
                best_solution, 
                method='L-BFGS-B', 
                bounds=bounds.T,
                options={'maxiter': min(100, self.budget - evaluations)}
            )

            evaluations += result.nfev

            # Update the best solution found
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Refine bounds based on current best solution
            bounds[0] = np.maximum(bounds[0], best_solution - 0.1 * np.ptp(bounds, axis=1))
            bounds[1] = np.minimum(bounds[1], best_solution + 0.1 * np.ptp(bounds, axis=1))

        return best_solution