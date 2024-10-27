import numpy as np
import scipy.optimize as optimize
import random

class LeverageGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.x_grad = np.zeros((dim, dim))
        self.x_hessian = np.zeros((dim, dim, dim, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a random point in the search space
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

            # Evaluate the function at the random point
            f = func(x)

            # Check if the current point is better than the best point found so far
            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # Compute the gradient of the function at the current point
            x_grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus_epsilon = x.copy()
                x_plus_epsilon[i] += 1e-6
                f_plus_epsilon = func(x_plus_epsilon)
                x_minus_epsilon = x.copy()
                x_minus_epsilon[i] -= 1e-6
                f_minus_epsilon = func(x_minus_epsilon)
                x_grad[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

            # Compute the Hessian of the function at the current point
            x_hessian = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] = (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] -= 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] -= 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] += (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)

            # Update the best point and the best gradient
            x_best_new = x + np.dot(self.x_grad, self.x_hessian[:, :, i, :]) * 1e-6
            f_best_new = func(x_best_new)
            if f_best_new < self.f_best:
                self.x_best = x_best_new
                self.f_best = f_best_new

            # Update the gradient and the Hessian
            self.x_grad = self.x_grad - np.dot(self.x_hessian[:, :, i, :], self.x_hessian[:, :, i, :]) * self.x_hessian[:, :, i, i]
            self.x_hessian[:, :, i, :] = self.x_hessian[:, :, i, :] / np.linalg.norm(self.x_hessian[:, :, i, :])

            # With a probability of 0.1, change the individual lines of the selected solution to refine its strategy
            if random.random() < 0.1:
                self.x_best = x + np.random.uniform(-1e-6, 1e-6, self.dim)
                self.f_best = func(self.x_best)

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = LeverageGradient(budget, dim)
alg()