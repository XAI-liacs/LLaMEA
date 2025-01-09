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

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, current, best):
        r = np.linalg.norm(best - current)
        beta = self.attractiveness(r)
        step = self.alpha * (np.random.rand(self.dim) - 0.5)
        return current + beta * (best - current) + step

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if intensities[i] > intensities[j]:
                        fireflies[i] = self.move_fireflies(fireflies[i], fireflies[j])
                        intensities[i] = func(fireflies[i])

        idx_min = np.argmin(intensities)
        self.f_opt = intensities[idx_min]
        self.x_opt = fireflies[idx_min]

        return self.f_opt, self.x_opt