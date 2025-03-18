import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(10 * self.dim, self.budget // 2)
        samples = self.uniform_sampling(bounds, num_initial_samples)
        
        best_sample = None
        best_value = float('inf')
        second_best_sample = None
        second_best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                second_best_sample = best_sample
                second_best_value = best_value
                best_value = value
                best_sample = sample
            elif value < second_best_value:
                second_best_value = value
                second_best_sample = sample

        remaining_budget = self.budget - num_initial_samples
        
        allocated_budget_1 = int(remaining_budget * np.random.uniform(0.4, 0.6))
        res1 = self.local_optimization(func, best_sample, bounds, allocated_budget_1)
        res2 = self.local_optimization(func, second_best_sample, bounds, remaining_budget - allocated_budget_1)
        
        perturbed_sample = best_sample + np.random.uniform(-0.01, 0.01, size=self.dim)  # New line for perturbation
        res3 = self.local_optimization(func, perturbed_sample, bounds, remaining_budget // 5)  # New optimization
        
        # Return the best result out of three local optimizations
        return min([res1, res2, res3], key=lambda x: x.fun)

    def uniform_sampling(self, bounds, num_samples):
        samples = []
        for _ in range(num_samples):
            sample = np.random.uniform([low for low, _ in bounds], 
                                       [high for _, high in bounds])
            samples.append(sample)
        return samples

    def local_optimization(self, func, initial_guess, bounds, budget):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': budget})
        return res