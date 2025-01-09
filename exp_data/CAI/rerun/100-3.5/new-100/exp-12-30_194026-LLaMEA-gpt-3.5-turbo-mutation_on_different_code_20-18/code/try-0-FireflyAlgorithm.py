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

    def attractiveness(self, x1, x2, func):
        return np.exp(-np.linalg.norm(func(x1) - func(x2)) / self.beta0)

    def move_firefly(self, x, best_x, func):
        r = np.random.uniform(-1, 1, size=self.dim)
        beta = self.beta0 * np.exp(-self.gamma * self.alpha)
        new_x = x + self.attractiveness(x, best_x, func) * (best_x - x) + beta * r
        return np.clip(new_x, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], func)

            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt