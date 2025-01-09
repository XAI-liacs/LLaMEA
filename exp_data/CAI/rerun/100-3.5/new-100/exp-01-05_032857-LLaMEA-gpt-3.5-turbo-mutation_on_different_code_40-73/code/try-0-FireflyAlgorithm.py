import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, xi, xj):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(xi - xj)**2)

    def move_firefly(self, x, best_x, func):
        beta = self.alpha * np.random.rand(self.dim)
        new_x = x + beta * (best_x - x) + self.alpha * (np.random.rand(self.dim) - 0.5)
        new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
        return new_x

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        f_values = np.apply_along_axis(func, 1, fireflies)

        for i in range(self.budget):
            for j in range(self.budget):
                if f_values[i] < f_values[j]:
                    fireflies[j] = self.move_firefly(fireflies[j], fireflies[i], func)
                    f_values[j] = func(fireflies[j])

        best_idx = np.argmin(f_values)
        self.f_opt = f_values[best_idx]
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt