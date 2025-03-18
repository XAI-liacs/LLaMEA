import numpy as np
from scipy.optimize import minimize

class DynamicBoundaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        remaining_budget = self.budget
        num_initial_guesses = max(2, self.budget // 15)
        
        # Generate initial guesses across the search space
        initial_guesses = [
            np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
            for _ in range(num_initial_guesses)
        ]
        
        best_solution = None
        best_value = float('inf')

        # Iteratively optimize from different starting points
        for init_guess in initial_guesses:
            if remaining_budget <= 0:
                break
            
            bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
            
            result = minimize(
                func,
                init_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': min(remaining_budget, 10)}  # Limit evaluations per run
            )
            
            remaining_budget -= result.nfev
            
            # Update the best known solution
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Dynamically adjust bounds based on the gradient information
            gradient = result.jac if result.jac is not None else np.zeros(self.dim)
            for i in range(self.dim):
                step_size = (func.bounds.ub[i] - func.bounds.lb[i]) * 0.05
                bounds[i] = (
                    max(bounds[i][0], best_solution[i] - step_size * np.sign(gradient[i])),
                    min(bounds[i][1], best_solution[i] + step_size * np.sign(gradient[i]))
                )
        
        return best_solution