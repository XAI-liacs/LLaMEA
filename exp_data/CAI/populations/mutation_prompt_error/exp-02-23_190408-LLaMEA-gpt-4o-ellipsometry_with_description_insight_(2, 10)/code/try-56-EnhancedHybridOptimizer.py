import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(10 * self.dim, self.budget // 2)  # Increased initial samples
        samples = self.uniform_sampling(bounds, num_initial_samples)
        
        best_sample = None
        best_value = float('inf')
        second_best_sample = None  # New line to track the second-best sample
        second_best_value = float('inf')  # New line to track the second-best value
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                second_best_sample = best_sample  # Update second-best sample
                second_best_value = best_value  # Update second-best value
                best_value = value
                best_sample = sample
            elif value < second_best_value:  # Condition to update second-best
                second_best_value = value
                second_best_sample = sample

        remaining_budget = self.budget - num_initial_samples
        
        # Use both best and second-best samples for local optimization
        allocated_budget_1 = int(remaining_budget * np.random.uniform(0.4, 0.6))  # Changed line to dynamically allocate budget
        method1 = 'Nelder-Mead' if allocated_budget_1 < 10 else 'L-BFGS-B'  # New line for adaptive method selection
        res1 = self.local_optimization(func, best_sample, bounds, allocated_budget_1, method1)
        res2 = self.local_optimization(func, second_best_sample, bounds, remaining_budget - allocated_budget_1, 'L-BFGS-B')
        
        # Return the best result out of two local optimizations
        return (res1.x, res1.fun) if res1.fun < res2.fun else (res2.x, res2.fun)

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.random.uniform([low for low, _ in bounds], 
                                       [high for _, high in bounds])
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, budget, method):  # Modified to include method parameter
        res = minimize(func, initial_guess, method=method, bounds=bounds, options={'maxfun': budget})
        return res