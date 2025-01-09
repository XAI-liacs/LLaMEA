import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.bounds = (-5.0, 5.0)
        self.particles = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])
                self.eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            if self.eval_count >= self.budget:
                break

            w = 0.5  # inertia weight
            c1, c2 = 1.5, 1.5  # cognitive and social coefficients

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component

                # Differential Evolution crossover
                if np.random.rand() < 0.5:
                    idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 2, replace=False)
                    donor_vector = self.personal_best_positions[idxs[0]] + 0.5 * (self.personal_best_positions[idxs[1]] - self.personal_best_positions[i])
                    trial_vector = np.copy(self.particles[i])
                    jrand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < 0.9 or j == jrand:
                            trial_vector[j] = donor_vector[j]

                    trial_vector = np.clip(trial_vector, *self.bounds)
                    trial_score = func(trial_vector)
                    self.eval_count += 1

                    if trial_score < score:
                        self.particles[i] = trial_vector
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector
                else:
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], *self.bounds)

        return self.global_best_position, self.global_best_score