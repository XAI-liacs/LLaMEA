import numpy as np
from scipy.optimize import minimize
from cma import CMAEvolutionStrategy

class PhotonicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Get bounds from func
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds = list(zip(lb, ub))
        
        # Number of CMA-ES samples
        num_cma_samples = min(5, self.budget // 20)
        
        # Generate initial CMA-ES samples
        cma_opts = {'bounds': [lb, ub], 'maxfevals': num_cma_samples}
        cma_es = CMAEvolutionStrategy(np.random.uniform(lb, ub), 0.25, cma_opts)
        
        best_solution = None
        best_value = float('inf')
        
        # CMA-ES for global search
        while not cma_es.stop():
            solutions = cma_es.ask()
            for solution in solutions:
                if self.evaluations >= self.budget:
                    break
                value = func(solution)
                self.evaluations += 1
                cma_es.tell([solution], [value])
                if value < best_value:
                    best_value = value
                    best_solution = solution
            
            if self.evaluations >= self.budget:
                break
        
        # Local optimization using L-BFGS-B starting from the best CMA-ES sample
        if best_solution is not None:
            res = minimize(func, best_solution, method='L-BFGS-B', bounds=bounds, options={'maxfun': self.budget - self.evaluations})
            self.evaluations += res.nfev
            if res.fun < best_value:
                best_value = res.fun
                best_solution = res.x
        
        return best_solution