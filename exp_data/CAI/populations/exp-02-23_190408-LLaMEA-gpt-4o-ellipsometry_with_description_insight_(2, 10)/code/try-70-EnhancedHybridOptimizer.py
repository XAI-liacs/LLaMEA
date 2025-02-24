import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = list(zip(func.bounds.lb, func.bounds.ub))
        num_initial_samples = min(10 * self.dim, self.budget // 2)
        samples = self.gradient_based_sampling(func, bounds, num_initial_samples)
        
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
        
        allocated_budget_1 = int(remaining_budget * 0.7)
        res1 = self.local_optimization(func, best_sample, bounds, allocated_budget_1)
        res2 = self.local_optimization(func, second_best_sample, bounds, remaining_budget - allocated_budget_1)
        
        return (res1.x, res1.fun) if res1.fun < res2.fun else (res2.x, res2.fun)

    def gradient_based_sampling(self, func, bounds, num_samples):
        samples = []
        adaptive_lr = 0.1
        for _ in range(num_samples):
            sample = np.random.uniform([low for low, _ in bounds], 
                                       [high for _, high in bounds])
            for _ in range(5):  # small gradient descent steps
                grad = self.estimate_gradient(func, sample)
                sample -= adaptive_lr * grad
                sample = np.clip(sample, [low for low, _ in bounds], [high for _, high in bounds])
            samples.append(sample)
        return samples

    def estimate_gradient(self, func, x, epsilon=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x[i] += epsilon
            f1 = func(x)
            x[i] -= 2 * epsilon
            f2 = func(x)
            grad[i] = (f1 - f2) / (2 * epsilon)
            x[i] += epsilon
        return grad

    def local_optimization(self, func, initial_guess, bounds, budget):
        res = minimize(func, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxfun': budget})
        return res