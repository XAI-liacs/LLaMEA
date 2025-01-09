import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.1, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.alpha * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        beta = self.beta0 * np.exp(-self.alpha * r)
        return x_i + beta * (x_j - x_i) + 0.01 * np.random.normal(size=self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(i+1, self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt