import numpy as np
from scipy.optimize import minimize

class DynamicAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        
        # Step 1: Initial Uniform Sampling for Diverse Starting Points
        num_initial_samples = min(3, self.budget // 3)
        initial_samples = np.random.uniform(
            low=func.bounds.lb,
            high=func.bounds.ub,
            size=(num_initial_samples, self.dim)
        )
        
        # Evaluate initial samples and store evaluations
        evaluations = [func(x) for x in initial_samples]
        remaining_budget = self.budget - num_initial_samples
        
        # Step 2: Dynamic Adaptive Sampling
        while remaining_budget > 0:
            # Select the best known sample
            best_idx = np.argmin(evaluations)
            best_sample = initial_samples[best_idx]
            
            # Adaptive Sampling around the best solution
            adaptive_samples = np.random.normal(
                loc=best_sample,
                scale=(func.bounds.ub - func.bounds.lb) * 0.1, 
                size=(num_initial_samples, self.dim)
            )
            adaptive_samples = np.clip(adaptive_samples, func.bounds.lb, func.bounds.ub)
            
            # Evaluate the adaptive samples
            adaptive_evaluations = [func(x) for x in adaptive_samples]
            evaluations.extend(adaptive_evaluations)
            initial_samples = np.vstack((initial_samples, adaptive_samples))
            remaining_budget -= num_initial_samples
            
            # Check if we have exhausted our budget
            if remaining_budget <= 0:
                break
        
        # Step 3: Dual-Phase Local Optimization
        best_idx = np.argmin(evaluations)
        best_sample = initial_samples[best_idx]
        
        def local_objective(x):
            return func(x)
        
        # Phase 1: Coarse Optimization with Nelder-Mead
        coarse_result = minimize(
            local_objective, 
            best_sample, 
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': remaining_budget // 2}
        )
        
        # Phase 2: Fine Optimization with BFGS
        fine_result = minimize(
            local_objective, 
            coarse_result.x, 
            method='BFGS',
            bounds=bounds,
            options={'maxiter': remaining_budget // 2}
        )
        
        return fine_result.x

# Example of usage:
# Assume func is a black-box function with attributes bounds.lb and bounds.ub
# optimizer = DynamicAdaptiveOptimizer(budget=100, dim=2)
# best_parameters = optimizer(func)