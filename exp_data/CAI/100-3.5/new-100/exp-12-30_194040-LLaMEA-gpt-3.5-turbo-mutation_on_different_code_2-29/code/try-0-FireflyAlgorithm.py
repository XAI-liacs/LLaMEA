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

    def attractiveness(self, x_i, x_j, f_i, f_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j))

    def move_firefly(self, x_i, x_j):
        r = np.linalg.norm(x_i - x_j)
        beta = self.beta0 * np.exp(-self.gamma * r)
        step = self.alpha * (np.random.rand(self.dim) - 0.5)
        x_new = x_i + beta * (x_j - x_i) + step
        return x_new

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.full(self.budget, np.Inf)

        for t in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if func(fireflies[i]) < func(fireflies[j]):
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])

            for i in range(self.budget):
                intensities[i] = func(fireflies[i])

            best_index = np.argmin(intensities)
            if intensities[best_index] < self.f_opt:
                self.f_opt = intensities[best_index]
                self.x_opt = fireflies[best_index]

        return self.f_opt, self.x_opt