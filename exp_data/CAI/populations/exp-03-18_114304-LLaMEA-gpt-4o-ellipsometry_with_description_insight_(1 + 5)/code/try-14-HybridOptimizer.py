import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        # Adaptive initial exploration with random sampling and gaussian perturbation
        num_samples = self.budget // 3
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        for i in range(num_samples):
            perturbed_sample = samples[i] + np.random.normal(0, 0.05, self.dim)
            perturbed_sample = np.clip(perturbed_sample, lb, ub)
            value = func(perturbed_sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = perturbed_sample

        # Gradient-enhanced local exploitation using BFGS
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value
            value = func(x)
            evaluations += 1
            return value

        # Dynamic bounds tightening
        dynamic_bounds = [(max(lb[i], best_solution[i] - 0.2), min(ub[i], best_solution[i] + 0.2)) for i in range(self.dim)]
        
        # Run BFGS using the perturbed best sample
        result = minimize(local_optimization, best_solution, method='BFGS',
                          bounds=dynamic_bounds,
                          options={'maxiter': self.budget - evaluations, 'disp': False})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution, best_value