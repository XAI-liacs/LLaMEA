import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        evaluations = 0

        # Improved exploration with Sobol sequence
        sampler = Sobol(d=self.dim, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(self.budget // 2)))
        samples = lb + samples * (ub - lb)
        for sample in samples:
            if evaluations >= self.budget:
                break
            value = func(sample)
            evaluations += 1
            if value < best_value:
                best_value = value
                best_solution = sample

        def local_optimization(x):
            nonlocal evaluations
            if evaluations >= self.budget:
                return best_value
            value = func(x)
            evaluations += 1
            return value

        # Switch to Trust-Region Reflective method
        result = minimize(local_optimization, best_solution, method='trust-constr',
                          bounds=[(lb[i], ub[i]) for i in range(self.dim)],
                          options={'maxiter': self.budget - evaluations, 'disp': False})

        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x

        return best_solution, best_value