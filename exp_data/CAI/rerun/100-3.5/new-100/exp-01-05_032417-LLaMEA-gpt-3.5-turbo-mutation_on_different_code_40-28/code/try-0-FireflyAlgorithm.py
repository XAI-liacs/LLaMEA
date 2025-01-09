import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        beta_i = self.beta * np.exp(-self.gamma * r) * (x_j - x_i) + self.alpha * np.random.uniform(-1, 1, self.dim)
        return x_i + beta_i

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt