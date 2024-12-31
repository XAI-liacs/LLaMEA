import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x1, x2):
        return np.exp(-self.beta * np.linalg.norm(x1 - x2))

    def move_firefly(self, x, dest, attractiveness):
        return x + self.alpha * (dest - x) * attractiveness + np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            f_current = func(fireflies[i])

            for j in range(self.budget):
                if func(fireflies[j]) < f_current:
                    attractiveness = self.attractiveness(fireflies[i], fireflies[j])
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], attractiveness)
                    f_current = func(fireflies[i])

            if f_current < self.f_opt:
                self.f_opt = f_current
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt