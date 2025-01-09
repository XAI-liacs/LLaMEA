import numpy as np

class SSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.iterations = budget // self.population_size

    def __call__(self, func):
        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma)
            v = np.random.normal(0, 1)
            step = u / abs(v) ** (1 / beta)
            return step
        
        def spider_move(spider_pos, best_pos):
            new_pos = spider_pos + levy_flight() * (best_pos - spider_pos)
            return np.clip(new_pos, -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness_values = np.apply_along_axis(func, 1, population)
        best_pos = population[np.argmin(fitness_values)]

        for _ in range(self.iterations):
            for i, spider_pos in enumerate(population):
                new_pos = spider_move(spider_pos, best_pos)
                new_fitness = func(new_pos)
                if new_fitness < fitness_values[i]:
                    population[i] = new_pos
                    fitness_values[i] = new_fitness
                    if new_fitness < func(best_pos):
                        best_pos = new_pos

        return best_pos