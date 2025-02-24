import numpy as np
from scipy.optimize import minimize

class GradientEnhancedBoundaryExplorer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Initial random samples for gradient estimation
        initial_samples = 5
        gradient_estimates = np.zeros((initial_samples, self.dim))
        values = np.zeros(initial_samples)
        
        for i in range(initial_samples):
            if evaluations >= self.budget:
                break
            
            x0 = np.random.uniform(lb, ub, self.dim)
            values[i] = func(x0)
            evaluations += 1

            if values[i] < best_value:
                best_solution = x0
                best_value = values[i]

            # Estimate gradients using finite differences
            for j in range(self.dim):
                perturb = np.zeros(self.dim)
                perturb[j] = 0.001 * (ub[j] - lb[j])
                gradient_estimates[i, j] = (func(x0 + perturb) - values[i]) / perturb[j]
                evaluations += 1

        # Determine promising search directions based on gradients
        avg_gradients = np.mean(gradient_estimates, axis=0)
        search_directions = np.sign(avg_gradients)
        
        while evaluations < self.budget:
            # Generate points by moving in the promising directions from the best solution
            trial_points = []
            step_size = 0.05 * (ub - lb)  # Small step size for boundary-focused exploration
            for i in range(self.dim):
                trial_points.append(best_solution + search_directions[i] * step_size)

            for point in trial_points:
                point = np.clip(point, lb, ub)
                res = minimize(func, point, method='L-BFGS-B', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
                evaluations += res.nfev

                if res.fun < best_value:
                    best_solution = res.x
                    best_value = res.fun

                if evaluations >= self.budget:
                    break
        
        return best_solution