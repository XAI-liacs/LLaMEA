import numpy as np

class GradientInspiredRandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.alpha = 0.01  # Initial step size for gradient-inspired update
        self.epsilon = 1e-8  # Small value to approximate gradient
        self.momentum = 0.9  # Momentum factor for update

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
        velocity = np.zeros_like(x)

        for _ in range(self.budget):
            gradient = self.estimate_gradient(func, x)
            velocity = self.momentum * velocity - self.alpha * gradient
            x_new = x + velocity
            x_new = np.clip(x_new, func_bounds.lb, func_bounds.ub)  # Ensure within bounds

            f_new = func(x_new)
            if f_new < self.f_opt:
                self.f_opt = f_new
                self.x_opt = x_new

            if np.random.rand() < 0.1:
                x = np.random.uniform(func_bounds.lb, func_bounds.ub, self.dim)
                self.alpha *= 0.9  # Decrease step size on random exploration
            else:
                x = x_new
                self.alpha *= 1.1  # Increase step size on successful steps

        return self.f_opt, self.x_opt