import numpy as np

class AdaptiveSGDOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _uniform_sample(self, bounds, num_samples):
        return np.array([np.random.uniform(low=bounds.lb[i], high=bounds.ub[i], size=num_samples) for i in range(self.dim)]).T

    def __call__(self, func):
        num_initial_samples = 5
        samples = self._uniform_sample(func.bounds, num_initial_samples)
        best_sample = samples[0]
        best_value = float('inf')
        
        for sample in samples:
            value = func(sample)
            if value < best_value:
                best_value = value
                best_sample = sample
                
        lr = 0.1
        remaining_budget = self.budget - num_initial_samples
        
        for _ in range(remaining_budget):
            gradient = self._approximate_gradient(func, best_sample)
            new_sample = best_sample - lr * gradient
            new_sample_clipped = np.clip(new_sample, func.bounds.lb, func.bounds.ub)
            
            new_value = func(new_sample_clipped)
            if new_value < best_value:
                best_value = new_value
                best_sample = new_sample_clipped
                lr *= 1.05  # slightly increase learning rate if improvement
            else:
                lr *= 0.5  # decrease learning rate if no improvement
        
        return best_sample

    def _approximate_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        fx = func(x)
        
        for i in range(self.dim):
            x_eps = np.copy(x)
            x_eps[i] += epsilon
            grad[i] = (func(x_eps) - fx) / epsilon
        
        return grad