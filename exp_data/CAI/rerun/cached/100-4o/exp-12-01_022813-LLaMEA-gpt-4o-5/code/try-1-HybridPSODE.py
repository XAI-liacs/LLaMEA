import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.c1 = 1.49  # cognitive coefficient
        self.c2 = 1.49  # social coefficient
        self.w = 0.729  # inertia weight
        self.F = 0.5    # DE scaling factor
        self.CR = 0.9   # DE crossover rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break

                # Update personal and global best
                score = func(self.positions[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
            
            # Update velocities and positions
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break

                # PSO velocity update
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                    self.c2 * r2 * (self.global_best_position - self.positions[i])
                )
                # Position update
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                # Differential Evolution mutation and crossover
                indices = np.random.choice(self.num_particles, 3, replace=False)
                a, b, c = self.positions[indices]
                mutant_vector = a + self.F * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(self.positions[i])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                # Evaluate trial vector
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.positions[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector

                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_score