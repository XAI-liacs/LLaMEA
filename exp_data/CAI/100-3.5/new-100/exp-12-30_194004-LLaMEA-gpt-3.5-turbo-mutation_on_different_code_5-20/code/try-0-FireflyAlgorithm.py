import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.alpha * r**2)

    def move_firefly(self, x_i, x_j, func):
        r = np.linalg.norm(x_i - x_j)
        beta = self.attractiveness(r)
        epsilon = np.random.uniform(-1, 1, size=self.dim)
        x_new = x_i + beta * (x_j - x_i) + 0.01 * epsilon
        f_new = func(x_new)
        return x_new, f_new

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[j]) < func(fireflies[i]):
                        fireflies[i], _ = self.move_firefly(fireflies[i], fireflies[j], func)

            best_index = np.argmin([func(x) for x in fireflies])
            if func(fireflies[best_index]) < self.f_opt:
                self.f_opt = func(fireflies[best_index])
                self.x_opt = fireflies[best_index]

        return self.f_opt, self.x_opt