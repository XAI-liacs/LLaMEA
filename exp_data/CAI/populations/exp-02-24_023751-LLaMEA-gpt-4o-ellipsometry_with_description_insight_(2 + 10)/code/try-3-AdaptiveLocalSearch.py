import numpy as np
from scipy.optimize import minimize

class AdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initial uniform random sampling
        num_initial_samples = min(self.budget // 2, 10 * self.dim)
        initial_solutions = np.random.uniform(lb, ub, (num_initial_samples, self.dim))
        
        # Evaluate initial solutions
        for solution in initial_solutions:
            value = func(solution)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_solution = solution

        # Adaptive local optimization using BFGS
        options = {'maxiter': self.budget, 'disp': False}
        def callback(xk):
            nonlocal best_value, best_solution
            current_value = func(xk)
            if current_value < best_value:
                best_value = current_value
                best_solution = xk

        # Refining bounds around the best solution
        exploration_factor = 0.15  # Changed exploration factor to be more aggressive
        bounds = [(max(lb[i], best_solution[i] - exploration_factor*(ub[i]-lb[i])), 
                   min(ub[i], best_solution[i] + exploration_factor*(ub[i]-lb[i]))) for i in range(self.dim)]

        result = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options=options, callback=callback)
        
        return best_solution if result.success else result.x