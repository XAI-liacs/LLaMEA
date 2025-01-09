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

    def move_fireflies(self, fireflies, func):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if func(fireflies[j]) < func(fireflies[i]):
                    r = np.linalg.norm(fireflies[j] - fireflies[i])
                    beta = self.attractiveness(r)
                    fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + self.alpha * np.random.normal(size=self.dim)

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, func)

        self.x_opt = fireflies[np.argmin([func(x) for x in fireflies])]
        self.f_opt = func(self.x_opt)

        return self.f_opt, self.x_opt