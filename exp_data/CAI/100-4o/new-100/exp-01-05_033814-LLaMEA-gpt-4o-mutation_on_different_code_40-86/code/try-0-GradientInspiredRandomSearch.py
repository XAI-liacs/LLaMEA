import numpy as np

class GradientInspiredRandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.alpha = 0.01  # Step size for gradient-inspired update
        self.epsilon = 1e-8  # Small value to approximate gradient

    def estimate_gradient(self, func, x):
        gradient = np.zeros_like(x)
        fx = func(x)
        for i in range(len(x)):
            x_eps = np.copy(x)
            x_eps[i] += self.epsilon
            gradient[i] = (func(x_eps) - fx) / self.epsilon
        return gradient

    def __call__(self, func):
        func_bounds = func.bounds
        x = np.random.uniform(func_bounds.lb, func_bounds.ub, self.dim)
        
        for _ in range(self.budget):
            gradient = self.estimate_gradient(func, x)
            x_new = x - self.alpha * gradient
            x_new = np.clip(x_new, func_bounds.lb, func_bounds.ub)  # Ensure within bounds

            f_new = func(x_new)
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new

            # Random exploration step
            if np.random.rand() < 0.1:  # 10% probability to explore randomly
                x = np.random.uniform(func_bounds.lb, func_bounds.ub, self.dim)
            else:
                x = x_new

        return self.f_opt, self.x_opt