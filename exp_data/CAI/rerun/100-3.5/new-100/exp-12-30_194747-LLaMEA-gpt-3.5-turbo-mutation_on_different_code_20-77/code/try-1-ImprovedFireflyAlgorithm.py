import numpy as np

class ImprovedFireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.alpha * np.linalg.norm(x - y))

    def move_firefly(self, x, y, func):
        new_x = x + self.attractiveness(y, x) * (y - x) + np.random.uniform(-1, 1, self.dim)
        new_x = np.clip(new_x, -5.0, 5.0)
        if func(new_x) < func(x):
            x = new_x
        return x

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], func)

            if func(fireflies[i]) < self.f_opt:
                self.f_opt = func(fireflies[i])
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt