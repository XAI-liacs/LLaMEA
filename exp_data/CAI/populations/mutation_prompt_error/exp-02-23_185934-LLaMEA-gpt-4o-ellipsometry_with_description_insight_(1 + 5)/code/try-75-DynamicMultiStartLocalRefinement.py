import numpy as np
from scipy.optimize import minimize

class DynamicMultiStartLocalRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.grid_samples = min(10, budget // 4)  # Adjusted initial sample size for diversified exploration

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Multi-start dynamic grid sampling
        grid_points = np.random.uniform(lb, ub, (self.grid_samples, self.dim))
        for i in range(self.grid_samples):
            x0 = grid_points[i]
            value = func(x0)
            self.evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = x0

            if self.evaluations >= self.budget:
                return best_solution

            # Hybrid local refinement with competitive selection
            if self.evaluations < self.budget * 0.75:
                res_bfgs = minimize(func, x0, method='L-BFGS-B', bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                                    options={'maxiter': self.budget // 20, 'gtol': 1e-6})
                if res_bfgs.fun < best_value:
                    best_value = res_bfgs.fun
                    best_solution = res_bfgs.x

                res_trust = minimize(func, x0, method='trust-constr', bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                                     options={'maxiter': self.budget // 20})
                if res_trust.fun < best_value:
                    best_value = res_trust.fun
                    best_solution = res_trust.x

                # Restart mechanism for exploring new regions
                if self.evaluations + 10 < self.budget:
                    x0_new = np.random.uniform(lb, ub, self.dim)
                    value_new = func(x0_new)
                    self.evaluations += 1
                    if value_new < best_value:
                        best_value = value_new
                        best_solution = x0_new

        # Final refinement with remaining budget
        remaining_budget = self.budget - self.evaluations
        if remaining_budget > 0:
            res = minimize(func, best_solution, method='trust-constr',
                           bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                           options={'maxiter': remaining_budget})
            if res.fun < best_value:
                best_solution = res.x

        return best_solution