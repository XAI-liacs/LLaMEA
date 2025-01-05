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

    def attractiveness(self, r, diversity):
        return self.beta0 * np.exp(-self.gamma * r**2) * (1 + 0.5 * diversity)

    def move_firefly(self, x, y, diversity):
        r = np.linalg.norm(x - y)
        beta = self.attractiveness(r, diversity)
        return x + beta * (y - x) + self.alpha * (np.random.rand(self.dim) - 0.5)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        intensities = np.array([func(ff) for ff in fireflies])

        for i in range(self.budget):
            diversity = np.std(fireflies, axis=0)
            for j in range(self.budget):
                if intensities[j] < intensities[i]:
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], diversity)
                    intensities[i] = func(fireflies[i])

        best_idx = np.argmin(intensities)
        self.f_opt = intensities[best_idx]
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt