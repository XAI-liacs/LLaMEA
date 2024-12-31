import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y, func):
        return np.exp(-self.beta * np.linalg.norm(x - y))

    def move_firefly(self, x, y, func):
        r = np.random.uniform(-1, 1, self.dim)
        return x + self.alpha * (y - x) + r

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))
        intensities = np.array([func(x) for x in fireflies])

        for i in range(self.budget):
            for j in range(self.budget):
                if intensities[i] > intensities[j]:
                    attractiveness_ij = self.attractiveness(fireflies[i], fireflies[j], func)
                    fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], func)
                    intensities[i] = func(fireflies[i])

        self.f_opt = np.min(intensities)
        self.x_opt = fireflies[np.argmin(intensities)]
        
        return self.f_opt, self.x_opt