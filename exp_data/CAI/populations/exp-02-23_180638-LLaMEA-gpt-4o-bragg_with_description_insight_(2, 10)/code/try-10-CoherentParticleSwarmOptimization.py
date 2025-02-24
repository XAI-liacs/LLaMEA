import numpy as np

class CoherentParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.population = None
        self.velocities = None
        self.lb = None
        self.ub = None
        self.fitness = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('-inf'))
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fit = func(self.population[i])
            if fit > self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fit
                self.personal_best_positions[i] = self.population[i].copy()
            if fit > self.global_best_fitness:
                self.global_best_fitness = fit
                self.global_best_position = self.population[i].copy()

    def update_velocities_and_positions(self):
        for i in range(self.population_size):
            cognitive_component = self.cognitive_coefficient * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.population[i])
            social_component = self.social_coefficient * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
            self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lb, self.ub)

    def periodicity_enforcement(self):
        for i in range(self.population_size):
            period = np.random.randint(1, self.dim // 2)
            for j in range(0, self.dim, period):
                averaged_value = np.mean(self.population[i][j:j+period])
                self.population[i][j:j+period] = averaged_value

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize()
        self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.update_velocities_and_positions()
            self.periodicity_enforcement()
            self.evaluate_population(func)
            evaluations += self.population_size

        return self.global_best_position