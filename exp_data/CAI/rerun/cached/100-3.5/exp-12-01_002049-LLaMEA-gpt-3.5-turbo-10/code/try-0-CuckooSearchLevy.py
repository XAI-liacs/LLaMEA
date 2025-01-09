import numpy as np

class CuckooSearchLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def levy_flight(self, size):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        nest = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.apply_along_axis(func, 1, nest)
        
        for _ in range(self.budget):
            new_nest = nest.copy()
            step_size = 0.01 * self.levy_flight((self.budget, self.dim))
            new_nest += step_size
            new_nest = np.clip(new_nest, -5.0, 5.0)
            new_fitness = np.apply_along_axis(func, 1, new_nest)
            idx = new_fitness < fitness
            nest[idx] = new_nest[idx]
            fitness[idx] = new_fitness[idx]
        
        return nest[np.argmin(fitness)]