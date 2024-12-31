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

    def attractiveness(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def move_firefly(self, x, y):
        beta = self.beta0 * np.exp(-self.alpha * i)
        new_x = x + beta * (y - x) + self.gamma * (np.random.rand(self.dim) - 0.5)
        return new_x

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    new_pos = self.move_firefly(fireflies[i], fireflies[j])
                    if func(new_pos) < func(fireflies[i]):
                        fireflies[i] = new_pos

            best_idx = np.argmin([func(x) for x in fireflies])
            if func(fireflies[best_idx]) < self.f_opt:
                self.f_opt = func(fireflies[best_idx])
                self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt