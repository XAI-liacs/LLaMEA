import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, r, light_intensity):
        return light_intensity * np.exp(-self.alpha * r**2)

    def move_firefly(self, x, light_intensity, fireflies):
        for firefly in fireflies:
            r = np.linalg.norm(x - firefly)
            if light_intensity < self.attractiveness(r, func(x)):
                x = x + self.alpha * (firefly - x) + np.random.uniform(-1, 1, self.dim)

        return x

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i] = self.move_firefly(fireflies[i], func(fireflies[i]), fireflies[j])

            if func(fireflies[i]) < self.f_opt:
                self.f_opt = func(fireflies[i])
                self.x_opt = fireflies[i]

        return self.f_opt, self.x_opt