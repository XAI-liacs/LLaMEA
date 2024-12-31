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

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    step = self.alpha * (np.random.rand(self.dim) - 0.5)
                    fireflies[i] += self.attractiveness(fireflies[i], fireflies[j]) * (fireflies[j] - fireflies[i]) + step

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)
            for i in range(len(fireflies)):
                f = func(fireflies[i])
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt
