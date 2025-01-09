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

    def attractiveness(self, light_intensity):
        return self.beta0 * np.exp(-self.gamma * light_intensity)

    def move_fireflies(self, fireflies, light_intensity):
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if light_intensity[j] < light_intensity[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = self.attractiveness(light_intensity[j])
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) / r**2

        return fireflies

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        light_intensity = np.array([func(x) for x in fireflies])

        for _ in range(self.budget):
            fireflies = self.move_fireflies(fireflies, light_intensity)
            light_intensity = np.array([func(x) for x in fireflies])

            idx = np.argmin(light_intensity)
            if light_intensity[idx] < self.f_opt:
                self.f_opt = light_intensity[idx]
                self.x_opt = fireflies[idx]

        return self.f_opt, self.x_opt