import numpy as np

class QIGSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.num_iterations = budget // self.num_particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def levy_flight():
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / beta)
            return 0.01 * step

        def fitness(x):
            return func(x)

        g_best = None
        g_best_fitness = np.inf

        for _ in range(self.num_iterations):
            particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
            for i in range(self.num_particles):
                particle = particles[i]
                cost = fitness(particle)
                if cost < g_best_fitness:
                    g_best = particle
                    g_best_fitness = cost
                step = levy_flight()
                new_particle = particle + step
                new_particle = np.clip(new_particle, self.lower_bound, self.upper_bound)
                new_cost = fitness(new_particle)
                if new_cost < cost:
                    particles[i] = new_particle

        return g_best