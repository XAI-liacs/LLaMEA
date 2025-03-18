import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from skopt import Optimizer

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')
        
        # Sobol sequence for initial sampling
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, self.budget // 10)
        samples = sobol_sampler.random_base2(m=int(np.log2(num_samples)))
        
        # Bayesian optimization setup for adaptive sampling
        space = [(lb[i], ub[i]) for i in range(self.dim)]
        bayes_optimizer = Optimizer(dimensions=space, n_initial_points=num_samples)
        
        for sample in samples:
            sample_scaled = lb + sample * (ub - lb)
            value = func(sample_scaled)
            bayes_optimizer.tell(sample_scaled, value)
            if value < best_value:
                best_value = value
                best_solution = sample_scaled
        
        # Adaptive sampling using Bayesian optimization
        remaining_budget = self.budget - num_samples
        for _ in range(remaining_budget):
            next_point = bayes_optimizer.ask()
            value = func(next_point)
            bayes_optimizer.tell(next_point, value)
            if value < best_value:
                best_value = value
                best_solution = next_point
        
        # Local optimization with L-BFGS-B for fine-tuning the best found solution
        def wrapped_func(x):
            return func(x)
        
        result = minimize(wrapped_func, best_solution, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': remaining_budget})
        
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x
        
        return best_solution