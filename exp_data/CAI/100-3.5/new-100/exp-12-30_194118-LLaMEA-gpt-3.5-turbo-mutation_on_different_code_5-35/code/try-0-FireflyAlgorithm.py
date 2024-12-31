import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.1, beta0=1.0, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j)**2)

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    beta = self.attractiveness(fireflies[i], fireflies[j])
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + self.alpha * (2 * np.random.rand(self.dim) - 1)

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)

        best_idx = np.argmin([func(x) for x in fireflies])
        self.f_opt = func(fireflies[best_idx])
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt