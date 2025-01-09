import numpy as np

class DynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_bound = (self.upper_bound - self.lower_bound) / 10.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.vel_bound, self.vel_bound, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations < self.budget:
                    score = func(self.population[i])
                    self.evaluations += 1

                    if score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = score
                        self.personal_best_positions[i] = self.population[i]

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.population[i]

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.population[i])
                new_velocity = self.w * self.velocities[i] + cognitive_velocity + social_velocity
                self.velocities[i] = np.clip(new_velocity, -self.vel_bound, self.vel_bound)
                new_position = self.population[i] + self.velocities[i]
                self.population[i] = np.clip(new_position, self.lower_bound, self.upper_bound)

        return self.global_best_position