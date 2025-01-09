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

    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def move_firefly(self, x, y, beta):
        r = np.random.uniform(-1, 1, size=self.dim)
        return x + beta * np.exp(-self.gamma * np.linalg.norm(x - y)**2) * r

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for _ in range(self.budget):
            for i, x in enumerate(fireflies):
                for j, y in enumerate(fireflies):
                    if func(x) < func(y):
                        beta = self.beta0 * np.exp(-self.alpha * np.linalg.norm(x - y)**2)
                        fireflies[i] = self.move_firefly(x, y, beta)

            for x in fireflies:
                f = func(x)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x
            
        return self.f_opt, self.x_opt