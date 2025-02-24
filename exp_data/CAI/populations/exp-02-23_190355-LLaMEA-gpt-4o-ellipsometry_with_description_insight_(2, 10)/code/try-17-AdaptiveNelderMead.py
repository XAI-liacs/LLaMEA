import numpy as np
from scipy.optimize import minimize

class AdaptiveNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        initial_guess = np.random.uniform(bounds[0], bounds[1], self.dim)

        # Calculate an approximate gradient direction using finite differences
        def approximate_gradient(x):
            eps = 1e-8
            grad = np.zeros_like(x)
            for i in range(self.dim):
                x_eps = x.copy()
                x_eps[i] += eps
                grad[i] = (func(x_eps) - func(x)) / eps
            return grad

        self.func_evaluations = 0
        def callback(x):
            self.func_evaluations += 1
            if self.func_evaluations >= self.budget:
                raise StopIteration("Budget exhausted")

        try:
            # Modify the initial_guess by incorporating a small step towards the gradient
            gradient_direction = approximate_gradient(initial_guess)
            initial_guess = initial_guess - 0.05 * gradient_direction  # Small step towards gradient

            result = minimize(func, initial_guess, method='Nelder-Mead', 
                              bounds=bounds.T, callback=callback, 
                              options={'maxiter': self.budget, 'disp': False})
            
            if result.success:
                return result.x
                
        except StopIteration:
            pass

        center = result.x
        bounds = np.clip(bounds, func.bounds.lb, func.bounds.ub)
        new_bounds = np.array([np.maximum(center - 0.1 * (bounds[1] - bounds[0]), bounds[0]),
                               np.minimum(center + 0.1 * (bounds[1] - bounds[0]), bounds[1])])
        
        try:
            result = minimize(func, result.x, method='Nelder-Mead', 
                              bounds=new_bounds.T, callback=callback, 
                              options={'maxiter': self.budget - self.func_evaluations, 'disp': False})
        except StopIteration:
            pass

        return result.x