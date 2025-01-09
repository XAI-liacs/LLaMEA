import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = np.copy(self.particles[i])

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(self.particles[i])

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i])) 
                # Adaptive velocity adjustment based on proximity to global_best_position
                self.velocities[i] *= (1 - 0.5 * (np.linalg.norm(self.particles[i] - self.global_best_position) / (self.upper_bound - self.lower_bound)))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

                # Differential Evolution Mutation (DE)
                indices = [idx for idx in range(self.num_particles) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.personal_best_positions[a] + self.f * (self.personal_best_positions[b] - self.personal_best_positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial_vector = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        trial_vector[j] = mutant_vector[j]

                # Selection
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < score:
                    self.particles[i] = trial_vector
                    self.personal_best_positions[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_score