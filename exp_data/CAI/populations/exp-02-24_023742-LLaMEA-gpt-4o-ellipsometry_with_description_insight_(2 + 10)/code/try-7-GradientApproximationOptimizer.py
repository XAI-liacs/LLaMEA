import numpy as np
from scipy.optimize import minimize

class GradientApproximationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = np.random.uniform(lb, ub, self.dim)

        def callback(xk):
            self.evals += 1

        def gradient_approx(func, x, epsilon=1e-5):
            grad = np.zeros_like(x)
            fx = func(x)
            for i in range(len(x)):
                x_eps = np.copy(x)
                x_eps[i] += epsilon
                grad[i] = (func(x_eps) - fx) / epsilon
            return grad

        def bounded_gradient_descent(func, x0, bounds, maxiter, learning_rate=0.01):
            x = x0
            for _ in range(maxiter):
                grad = gradient_approx(func, x)
                x = x - learning_rate * grad
                x = np.clip(x, bounds.lb, bounds.ub)
                if self.evals >= self.budget:
                    break
            return x, func(x)

        best_x, best_f = x0, float('inf')
        remaining_budget = self.budget

        while remaining_budget > 0:
            maxiter = min(remaining_budget, 100)
            x_opt, f_opt = bounded_gradient_descent(func, x0, func.bounds, maxiter)
            
            if f_opt < best_f:
                best_x, best_f = x_opt, f_opt
                x0 = best_x
                r = 0.1 * (ub - lb)
                lb, ub = np.maximum(func.bounds.lb, best_x - r), np.minimum(func.bounds.ub, best_x + r)

            remaining_budget -= maxiter
            if self.evals >= self.budget:
                break

        return best_x