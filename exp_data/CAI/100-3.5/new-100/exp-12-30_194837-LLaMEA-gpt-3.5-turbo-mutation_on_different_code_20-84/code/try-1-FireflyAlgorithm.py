import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=2.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, func):
        return 1.0 / (1 + func(x))

    def move(self, x, x_neighbor, alpha, gamma, iteration):
        beta = self.beta0 * np.exp(-gamma * np.linalg.norm(x - x_neighbor))
        return x + alpha * (x_neighbor - x) + beta * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub)
        for i in range(self.budget):
            for j in range(self.budget):
                x_neighbor = np.random.uniform(func.bounds.lb, func.bounds.ub)
                if func(x_neighbor) < func(x):
                    x = self.move(x, x_neighbor, self.alpha * (1 - i / self.budget), self.gamma / (1 + i), i)
            if func(x) < self.f_opt:
                self.f_opt = func(x)
                self.x_opt = x

        return self.f_opt, self.x_opt