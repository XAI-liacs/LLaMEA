import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10, alpha=0.5, beta0=1.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta0 = beta0
        self.f_opt = np.Inf
        self.x_opt = None

    def attractiveness(self, x, y):
        return self.beta0 * np.exp(-self.alpha * np.linalg.norm(x - y))

    def move_firefly(self, x, target, best):
        beta = self.attractiveness(best, target)
        new_x = x + beta * (target - x) + 0.01 * np.random.normal(0, 1, self.dim)
        return np.clip(new_x, -5.0, 5.0)

    def __call__(self, func):
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            f_vals = np.array([func(x) for x in fireflies])
            best_idx = np.argmin(f_vals)
            best = fireflies[best_idx]

            for j in range(self.budget):
                for k in range(self.budget):
                    if f_vals[j] < f_vals[k]:
                        fireflies[j] = self.move_firefly(fireflies[j], fireflies[k], best)

            if f_vals[best_idx] < self.f_opt:
                self.f_opt = f_vals[best_idx]
                self.x_opt = fireflies[best_idx]

        return self.f_opt, self.x_opt