import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.differential_weight = 0.8
        self.crossover_probability = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.population[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]

                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions[i] - self.population[i])
                social_velocity = self.social_constant * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                )
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant_vector = x1 + self.differential_weight * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_probability
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

            if evaluations >= self.budget:
                break

        return self.global_best_position, self.global_best_score