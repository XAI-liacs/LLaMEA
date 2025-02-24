import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from skopt import Optimizer

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        num_initial_samples = min(self.budget // 2, max(10, self.dim))
        
        sobol_sampler = qmc.Sobol(d=self.dim, scramble=True)
        sobol_samples = qmc.scale(sobol_sampler.random(num_initial_samples // 2), lb, ub)
        random_samples = np.random.uniform(lb, ub, (num_initial_samples // 2, self.dim))
        initial_samples = np.vstack((random_samples, sobol_samples))
        
        best_solution = None
        best_value = float('inf')
        
        for sample in initial_samples:
            result = minimize(func, sample, method='L-BFGS-B', bounds=np.array(list(zip(lb, ub))))
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
        
        remaining_budget = self.budget - num_initial_samples
        opt = Optimizer(dimensions=[(lb[i], ub[i]) for i in range(self.dim)], n_initial_points=0)
        
        while remaining_budget > 0:
            current_bounds = [(max(lb[i], best_solution[i] - 0.1 * (ub[i] - lb[i])), min(ub[i], best_solution[i] + 0.1 * (ub[i] - lb[i]))) for i in range(self.dim)]
            opt.tell(initial_samples.tolist(), [func(x) for x in initial_samples])
            next_points = opt.ask(n_points=1)
            
            for next_point in next_points:
                result = minimize(func, next_point, method='L-BFGS-B', bounds=current_bounds)
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
            
            remaining_budget -= 1
        
        return best_solution