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

    def attractiveness(self, x_i, x_j, f_i, f_j):
        return self.beta0 * np.exp(-self.gamma * np.linalg.norm(x_i - x_j)**2)

    def move_firefly(self, x_i, x_j):
        return x_i + self.alpha * (x_j - x_i) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[j] < intensities[i]:
                    attractiveness_ij = self.attractiveness(fireflies[i], fireflies[j], intensities[i], intensities[j])
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j]) if attractiveness_ij > np.random.rand() else fireflies[i]
                    intensities[i] = func(fireflies[i])

            best_idx = np.argmin(intensities)
            if intensities[best_idx] < self.f_opt:
                self.f_opt = intensities[best_idx]
                self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt