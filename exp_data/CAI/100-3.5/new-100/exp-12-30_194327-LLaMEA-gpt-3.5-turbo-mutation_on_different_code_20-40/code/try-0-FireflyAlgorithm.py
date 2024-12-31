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

    def attractiveness(self, x_i, x_j):
        return np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def move_fireflies(self, x, f, func):
        for i in range(self.budget):
            for j in range(self.budget):
                if f[j] < f[i]:
                    beta = self.beta0 * np.exp(-self.alpha * np.square(np.linalg.norm(x[j] - x[i])))
                    x[i] += beta * (x[j] - x[i]) + np.random.uniform(-1, 1, self.dim)
                    x[i] = np.clip(x[i], func.bounds.lb, func.bounds.ub)
                    f[i] = func(x[i])
                    if f[i] < self.f_opt:
                        self.f_opt = f[i]
                        self.x_opt = x[i]

        return self.f_opt, self.x_opt

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        f = np.array([func(xi) for xi in x])

        return self.move_fireflies(x, f, func)