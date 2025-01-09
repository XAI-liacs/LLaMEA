import numpy as np

class LevyFlightPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w = 0.9   # Initial inertia weight
        self.velocity_clamp = (-1, 1)  # Velocity clamp
        self.alpha = 0.05  # Step size for Levy flight

    def levy_flight(self, step):
        beta = 1.5
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=step.shape)
        v = np.random.normal(0, 1, size=step.shape)
        step = u / np.abs(v) ** (1 / beta)
        return self.alpha * step

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best[i] = self.population[i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.population[i].copy()

    def update_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_best - self.population)
        social_velocity = self.c2 * r2 * (self.global_best - self.population)
        self.velocities = self.w * self.velocities + cognitive_velocity + social_velocity + self.levy_flight(self.velocities)
        self.velocities = np.clip(self.velocities, *self.velocity_clamp)
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, lb, ub)

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.w = 0.9 * np.exp(-3 * evaluations / self.budget)  # Exponential decay for inertia weight
            self.c2 = 1.5 + (0.1 * evaluations / self.budget)  # Adaptive adjustment of c2
            self.update_particles(func.bounds)

        return self.global_best, self.global_best_fitness