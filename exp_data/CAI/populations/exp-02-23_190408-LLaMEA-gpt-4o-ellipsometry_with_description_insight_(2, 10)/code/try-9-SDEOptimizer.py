import numpy as np

class SDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _random_sample(self, bounds):
        return np.array([np.random.uniform(low=bounds.lb[i], high=bounds.ub[i]) for i in range(self.dim)])

    def _estimate_gradient(self, func, x, epsilon=1e-5):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_plus = np.copy(x)
            x_plus[i] += epsilon
            grad[i] = (func(x_plus) - func(x)) / epsilon
        return grad

    def __call__(self, func):
        x = self._random_sample(func.bounds)
        learning_rate = 0.1
        dual_avg = np.zeros(self.dim)
        
        for _ in range(self.budget):
            grad = self._estimate_gradient(func, x)
            dual_avg += grad
            x -= learning_rate * dual_avg / (_ + 1)
            
            # Projecting back to bounds
            x = np.maximum(func.bounds.lb, np.minimum(func.bounds.ub, x))
        
        return x