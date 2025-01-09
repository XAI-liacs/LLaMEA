import numpy as np

class HybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.7   # inertia weight
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                current_score = func(self.population[i])
                self.evaluations += 1

                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.population[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.population[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = self.w * self.velocities[i]
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = inertia + cognitive + social

                # Apply adaptive differential mutation
                mutation_factor = 0.5 + 0.5 * np.random.rand()
                differential_mutation = mutation_factor * (self.personal_best_positions[i] - self.global_best_position)
                self.velocities[i] += differential_mutation

                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        return self.global_best_score, self.global_best_position