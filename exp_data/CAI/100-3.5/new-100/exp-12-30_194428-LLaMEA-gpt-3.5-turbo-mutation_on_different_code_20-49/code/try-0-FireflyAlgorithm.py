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
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x - y))

    def move_firefly(self, x, y):
        r = np.linalg.norm(x - y)
        beta = self.beta0 * np.exp(-self.gamma * r**2)
        return x + self.alpha * (beta * (y - x)) + 0.01 * np.random.randn(self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if intensities[j] < intensities[i]:
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])
                        intensities[i] = func(fireflies[i])

        best_idx = np.argmin(intensities)
        self.f_opt = intensities[best_idx]
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt