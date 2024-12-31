import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.01):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, x, f, func):
        for j in range(self.budget):
            for i in range(self.budget):
                if f[j] > f[i]:
                    r = np.linalg.norm(x[i] - x[j])
                    beta = self.attractiveness(r)
                    x[j] += beta * (x[i] - x[j]) + self.alpha * np.random.uniform(-1, 1, self.dim)
                    x[j] = np.clip(x[j], func.bounds.lb, func.bounds.ub)
                    f[j] = func(x[j])
                    if f[j] < self.f_opt:
                        self.f_opt = f[j]
                        self.x_opt = x[j]

        return self.f_opt, self.x_opt

    def __call__(self, func):
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        f = np.array([func(xi) for xi in x])

        self.f_opt, self.x_opt = self.move_fireflies(x, f, func)

        return self.f_opt, self.x_opt