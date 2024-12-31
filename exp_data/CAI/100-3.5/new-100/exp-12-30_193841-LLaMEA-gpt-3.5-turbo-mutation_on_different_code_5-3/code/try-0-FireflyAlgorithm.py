import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.1):
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
        for i in range(self.budget):
            for j in range(self.budget):
                if f[j] < f[i]:
                    r = np.linalg.norm(x[i] - x[j])
                    beta = self.attractiveness(r)
                    x[i] = x[i] + self.alpha * (x[j] - x[i]) * beta
                    f[i] = func(x[i])

        return x, f

    def __call__(self, func):
        x = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        f = np.array([func(x[i]) for i in range(self.budget)])

        x, f = self.move_fireflies(x, f, func)

        idx = np.argmin(f)
        self.f_opt = f[idx]
        self.x_opt = x[idx]

        return self.f_opt, self.x_opt