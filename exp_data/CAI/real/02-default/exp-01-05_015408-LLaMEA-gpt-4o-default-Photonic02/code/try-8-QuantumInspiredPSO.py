import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.phi = 1.61803398875 # Golden ratio for adaptive neighborhood
        self.inertia_weight = 0.7
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(ind) for ind in particles])
        best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity with quantum behavior
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i])
                    + self.social_coeff * r2 * (global_best_position - particles[i])
                )
                # Apply quantum-inspired update
                velocities[i] = (velocities[i] + np.sin(velocities[i]) * self.phi) / 2

                # Move the particle
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particle
                score = func(particles[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = score

                # Update global best
                if score < personal_best_scores[best_index]:
                    global_best_position = particles[i]

                if evaluations >= self.budget:
                    break

        return global_best_position, personal_best_scores[best_index]