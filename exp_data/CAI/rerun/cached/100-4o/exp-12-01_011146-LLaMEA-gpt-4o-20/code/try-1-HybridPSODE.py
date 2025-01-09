import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover probability

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Particle Swarm Optimization step
            for i in range(self.population_size):
                current_score = func(self.population[i])
                evaluations += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.population[i]
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.population[i]

            for i in range(self.population_size):
                inertia = 0.5
                cognitive = 2 * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.population[i])
                social = 2 * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                self.velocities[i] = inertia * self.velocities[i] + cognitive + social
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.bounds[0], self.bounds[1])

            if evaluations >= self.budget:
                break

            # Differential Evolution step
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.f * (b - c), self.bounds[0], self.bounds[1])

                trial_vector = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.cr
                trial_vector[cross_points] = mutant_vector[cross_points]

                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

                if trial_score < func(self.population[i]):
                    self.population[i] = trial_vector

        return self.global_best_position