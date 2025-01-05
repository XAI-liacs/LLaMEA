import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0, gamma=0.01, elite_ratio=0.1):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.elite_ratio = elite_ratio
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return self.beta0 * np.exp(-self.gamma * r**2)

    def move_fireflies(self, current, best):
        r = np.linalg.norm(current - best)
        beta = self.attractiveness(r)
        step = beta * (current - best) + self.alpha * (np.random.rand(self.dim) - 0.5)
        return current + step

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[i] > intensities[j]:
                    fireflies[i] = self.move_fireflies(fireflies[i], fireflies[j])
                    intensities[i] = func(fireflies[i])

        elite_count = int(self.elite_ratio * self.budget)
        elite_indices = np.argsort(intensities)[:elite_count]

        for elite_idx in elite_indices:
            for i in range(self.budget):
                if intensities[i] > intensities[elite_idx]:
                    fireflies[i] = self.move_fireflies(fireflies[i], fireflies[elite_idx])
                    intensities[i] = func(fireflies[i])

        best_index = np.argmin(intensities)
        self.f_opt = intensities[best_index]
        self.x_opt = fireflies[best_index]

        return self.f_opt, self.x_opt