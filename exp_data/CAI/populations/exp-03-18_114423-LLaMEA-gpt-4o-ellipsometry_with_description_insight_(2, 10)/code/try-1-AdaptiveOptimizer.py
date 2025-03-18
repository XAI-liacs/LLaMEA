import numpy as np
from scipy.optimize import minimize

class AdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.history = []

    def __call__(self, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        x0 = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub, size=self.dim)
        
        def callback(xk):
            if self.evaluations >= self.budget:
                return True
            self.evaluations += 1
            self.history.append((xk.copy(), func(xk)))
        
        options = {'disp': False, 'maxiter': self.budget}
        
        while self.evaluations < self.budget:
            result = minimize(func, x0, method='L-BFGS-B', bounds=bounds, callback=callback, options=options)
            if self.evaluations >= self.budget:
                break
            
            # Update bounds based on the successful iterations
            best_position = result.x
            bounds = [(max(func.bounds.lb[i], best_position[i] - 0.1*(func.bounds.ub[i] - func.bounds.lb[i])),
                       min(func.bounds.ub[i], best_position[i] + 0.1*(func.bounds.ub[i] - func.bounds.lb[i])))
                      for i in range(self.dim)]
            
            x0 = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=self.dim)

        best_solution = min(self.history, key=lambda x: x[1])[0]
        return best_solution