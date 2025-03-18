import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0
        
        # Initial exploration with adaptive sampling
        adaptive_samples = int(self.budget * 0.4)
        samples = np.random.uniform(lb, ub, (adaptive_samples, self.dim))
        for sample in samples:
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        # Dual-phase BFGS exploration
        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value  # Return the best value if budget is exhausted
            value = func(x)
            evaluations += 1
            return value

        # Dynamic bounds for dual exploration phases
        dynamic_bounds_phase_1 = [(max(lb[i], best_solution[i] - 0.2), min(ub[i], best_solution[i] + 0.2)) for i in range(self.dim)]
        
        # Phase 1 of BFGS
        result_phase_1 = minimize(local_optimization, best_solution, method='BFGS',
                                  bounds=dynamic_bounds_phase_1,
                                  options={'maxiter': (self.budget - evaluations) // 2, 'disp': False})

        if result_phase_1.fun < best_value:
            best_value = result_phase_1.fun
            best_solution = result_phase_1.x

        # Phase 2 of BFGS with tighter bounds
        dynamic_bounds_phase_2 = [(max(lb[i], best_solution[i] - 0.05), min(ub[i], best_solution[i] + 0.05)) for i in range(self.dim)]
        
        result_phase_2 = minimize(local_optimization, best_solution, method='BFGS',
                                  bounds=dynamic_bounds_phase_2,
                                  options={'maxiter': (self.budget - evaluations), 'disp': False})

        if result_phase_2.fun < best_value:
            best_value = result_phase_2.fun
            best_solution = result_phase_2.x

        return best_solution, best_value