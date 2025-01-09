import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x_i, x_j, f_i, f_j):
        r = np.linalg.norm(x_i - x_j)
        return self.beta0 * np.exp(-self.gamma * r ** 2) * (f_i - f_j)

    def move_firefly(self, x_i, f_i, x_j, f_j):
        beta = self.attractiveness(x_i, x_j, f_i, f_j)
        x_i += self.alpha * (x_j - x_i) + beta * (np.random.rand(self.dim) - 0.5)
        return x_i

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        f_values = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if f_values[j] < f_values[i]:
                    fireflies[i] = self.move_firefly(fireflies[i], f_values[i], fireflies[j], f_values[j])
                    f_values[i] = func(fireflies[i])

        best_index = np.argmin(f_values)
        self.f_opt = f_values[best_index]
        self.x_opt = fireflies[best_index]

        return self.f_opt, self.x_opt