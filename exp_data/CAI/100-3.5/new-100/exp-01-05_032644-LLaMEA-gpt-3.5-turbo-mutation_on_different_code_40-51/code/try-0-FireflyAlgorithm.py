import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.alpha * r**2)

    def move_firefly(self, x, y, attractiveness):
        beta = self.attractiveness(np.linalg.norm(x - y))
        return x + beta * (y - x) + 0.01 * np.random.randn(self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], self.attractiveness(np.linalg.norm(fireflies[i] - fireflies[j])))

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt