import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.population_size, np.inf)
        self.best_global_position = None
        self.best_global_score = np.inf
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                score = func(self.particles[i])
                self.evaluations += 1
                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.particles[i]
                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.particles[i]

            if self.evaluations >= self.budget:
                break

            # Update velocity and position
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_constant * r1 * (self.best_personal_positions[i] - self.particles[i])
                social_velocity = self.social_constant * r2 * (self.best_global_position - self.particles[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Adaptive inertia weight update
            self.inertia_weight = 0.9 - 0.5 * (self.evaluations / self.budget)

            # Apply Differential Evolution strategy
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_probability:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = self.particles[np.random.choice(indices, 3, replace=False)]
                    trial_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                    # Evaluate trial vector
                    trial_score = func(trial_vector)
                    self.evaluations += 1
                    if trial_score < self.best_personal_scores[i]:
                        self.best_personal_scores[i] = trial_score
                        self.best_personal_positions[i] = trial_vector
                        self.particles[i] = trial_vector
                        if trial_score < self.best_global_score:
                            self.best_global_score = trial_score
                            self.best_global_position = trial_vector

                    if self.evaluations >= self.budget:
                        break

        return self.best_global_score, self.best_global_position