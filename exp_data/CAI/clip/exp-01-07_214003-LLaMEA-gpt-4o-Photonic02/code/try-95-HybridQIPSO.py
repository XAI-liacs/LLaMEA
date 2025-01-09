import numpy as np

class HybridQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.inertia_weight = 0.8
        self.mutation_rate = 0.1
        self.velocity_clamp = (-1, 1)

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_bests = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_bests[i] = self.population[i].copy()
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = self.population[i].copy()

    def update_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_bests - self.population)
        social_velocity = self.c2 * r2 * (self.global_best - self.population)
        self.velocities = self.inertia_weight * self.velocities + cognitive_velocity + social_velocity
        self.velocities = np.clip(self.velocities, *self.velocity_clamp)
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, lb, ub)

    def quantum_inspired_mutation(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                quantum_bit = np.random.uniform(lb, ub, self.dim)
                self.population[i] = self.population[i] + np.sin(quantum_bit - self.population[i])
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.update_particles(func.bounds)
            evaluations += self.population_size
            if evaluations >= self.budget:
                break
            self.quantum_inspired_mutation(func.bounds)
            evaluations += self.population_size

        return self.global_best, self.global_best_fitness