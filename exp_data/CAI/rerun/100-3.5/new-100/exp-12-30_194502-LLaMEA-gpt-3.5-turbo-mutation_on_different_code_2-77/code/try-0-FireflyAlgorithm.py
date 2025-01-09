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

    def attractiveness(self, light_intensity):
        return self.beta0 * np.exp(-self.gamma * light_intensity)

    def move_firefly(self, x_current, x_firefly, attractiveness):
        return x_firefly + self.alpha * (x_current - x_firefly) + attractiveness * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        light_intensities = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if light_intensities[i] > light_intensities[j]:
                    attractiveness_ij = self.attractiveness(light_intensities[i])
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], attractiveness_ij)
                    light_intensities[i] = func(fireflies[i])

        best_idx = np.argmin(light_intensities)
        self.f_opt = light_intensities[best_idx]
        self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt