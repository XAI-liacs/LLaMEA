import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j):
        return np.exp(-self.beta * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j):
        r = np.random.rand(self.dim)
        x_new = x_i + self.alpha * (x_j - x_i) + self.attractiveness(x_i, x_j) * (r - 0.5)
        return np.clip(x_new, -5.0, 5.0)

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])

        best_idx = np.argmin([func(f) for f in fireflies])
        self.f_opt = func(fireflies[best_idx])
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt