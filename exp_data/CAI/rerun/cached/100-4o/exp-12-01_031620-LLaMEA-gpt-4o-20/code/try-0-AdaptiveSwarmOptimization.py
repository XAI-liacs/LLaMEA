import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.4
        self.social_weight = 1.4
        self.velocities = np.zeros((self.population_size, self.dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.global_best_position = None
        self.best_fitness = float('inf')

    def evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.positions)
        return fitness

    def update_velocities_positions(self):
        r1, r2 = np.random.rand(2)
        for i in range(self.population_size):
            cognitive_component = self.cognitive_weight * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.social_weight * r2 * (self.global_best_position - self.positions[i])
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  cognitive_component + social_component)
            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if fitness[i] < self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.global_best_position = self.positions[i]

                if fitness[i] < np.apply_along_axis(func, 1, self.personal_best_positions[[i]])[0]:
                    self.personal_best_positions[i] = self.positions[i]
            
            self.update_velocities_positions()

        return self.global_best_position, self.best_fitness