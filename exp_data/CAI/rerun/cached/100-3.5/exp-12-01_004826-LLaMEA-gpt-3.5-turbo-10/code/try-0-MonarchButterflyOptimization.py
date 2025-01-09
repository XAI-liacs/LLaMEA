import numpy as np

class MonarchButterflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.max_step = 0.2
        self.alpha = 0.9

    def levy_flight(self):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, sigma_v, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)

        for _ in range(self.budget - self.pop_size):
            step = self.levy_flight()
            new_pos = pop[best_idx] + step * self.max_step
            new_pos = np.clip(new_pos, -5.0, 5.0)
            new_fit = func(new_pos)

            if new_fit < fitness[best_idx]:
                pop[best_idx] = new_pos
                fitness[best_idx] = new_fit

            best_idx = np.argmin(fitness)

        return pop[best_idx]