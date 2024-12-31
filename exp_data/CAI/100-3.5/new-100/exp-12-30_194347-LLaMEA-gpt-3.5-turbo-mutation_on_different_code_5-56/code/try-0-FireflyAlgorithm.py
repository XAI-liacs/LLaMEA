import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return np.exp(-self.gamma * r)

    def move_firefly(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        beta_rand = self.beta * (np.random.rand(self.dim) - 0.5)
        return x_i + self.attractiveness(r) * (x_j - x_i) + self.alpha * beta_rand

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[i]) < func(fireflies[j]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt