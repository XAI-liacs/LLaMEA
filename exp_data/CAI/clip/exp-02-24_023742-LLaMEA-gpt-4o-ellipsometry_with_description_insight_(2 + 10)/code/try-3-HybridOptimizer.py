import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define bounds and initialize Sobol sequence
        lb, ub = func.bounds.lb, func.bounds.ub
        sobol = Sobol(d=self.dim, scramble=True)
        
        # Initial sampling with Sobol sequence
        n_init = min(self.budget // 2, 50)  # Use half of the budget or 50 samples
        samples = sobol.random_base2(m=int(np.log2(n_init)))
        scaled_samples = lb + samples * (ub - lb)

        # Evaluate initial samples
        best_x = None
        best_f = float('inf')
        evaluations = 0

        for x in scaled_samples:
            f_val = func(x)
            evaluations += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x

        # Local optimization with BFGS
        def objective(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                raise Exception("Budget exceeded")
            evaluations += 1
            return func(x)

        result = minimize(objective, best_x, method='L-BFGS-B', bounds=[(l, u) for l, u in zip(lb, ub)])

        return result.x

# Usage example:
# optimizer = HybridOptimizer(budget=100, dim=2)
# best_parameters = optimizer(my_black_box_function)