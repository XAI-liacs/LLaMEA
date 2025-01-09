import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta_min=0.2, gamma=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r):
        return np.exp(-self.gamma * r**2)

    def move_firefly(self, x, x_best):
        r = np.linalg.norm(x - x_best)
        beta = self.beta_min * np.exp(-self.alpha * r**2)
        x = x + beta * (x_best - x) + 0.01 * np.random.uniform(-1, 1, size=self.dim)
        return np.clip(x, -5.0, 5.0)

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        intensities = np.array([func(firefly) for firefly in fireflies])

        for _ in range(self.budget):
            for i in range(len(fireflies)):
                for j in range(len(fireflies)):
                    if intensities[j] < intensities[i]:
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j])
                        intensities[i] = func(fireflies[i])

            idx = np.argmin(intensities)
            if intensities[idx] < self.f_opt:
                self.f_opt = intensities[idx]
                self.x_opt = fireflies[idx]

        return self.f_opt, self.x_opt