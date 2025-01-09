import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 30
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.array([float('inf')] * self.num_particles)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.global_best_score = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(self.particles[i])
                self.evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

                if self.evaluations >= self.budget:
                    break

            # Quantum-inspired superposition principle for dynamic velocity update
            w = 0.5 + (np.random.rand() / 2)  # inertia weight
            c1 = 2.0  # cognitive component
            c2 = 2.0  # social component

            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_velocity = c2 * r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity

                # Update particle position with superposition principle
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score