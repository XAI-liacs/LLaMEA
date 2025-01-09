import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.2, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return np.exp(-self.beta0 * np.linalg.norm(x - y))

    def move_firefly(self, x_i, x_j, func):
        beta = self.beta0 * np.exp(-self.alpha*i)
        new_x_i = x_i + beta * (x_j - x_i) + 0.01 * np.random.normal(0, 1, self.dim)
        new_f_i = func(new_x_i)
        return new_x_i, new_f_i

    def __call__(self, func):
        fireflies = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for i in range(self.budget):
            for j in range(self.budget):
                if func(fireflies[j]) < func(fireflies[i]):
                    fireflies[i], _ = self.move_firefly(fireflies[i], fireflies[j], func)
            
            f = func(fireflies[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = fireflies[i]
            
        return self.f_opt, self.x_opt