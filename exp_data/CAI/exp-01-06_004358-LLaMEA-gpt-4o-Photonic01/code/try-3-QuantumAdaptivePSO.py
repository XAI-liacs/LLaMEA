import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(self.dim))
        self.inertia_weight = 0.9  # Changed initial inertia weight
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.global_best_position = None
        self.global_best_value = float('inf')

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.full(self.population_size, float('inf'))
        
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                fitness = func(particles[i])
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = particles[i].copy()

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = particles[i].copy()

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_term = self.cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i])
                social_term = self.social_coefficient * r2 * (self.global_best_position - particles[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_term + social_term
                particles[i] += velocities[i]
                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lb, ub)

            # Adaptive mutation
            mutation_prob = 1.0 / (1.0 + np.exp(-0.2 * (evaluations / self.budget - 0.5)))
            for i in range(self.population_size):
                if np.random.rand() < mutation_prob:
                    mutation_vector = np.random.uniform(-0.1, 0.1, self.dim)
                    particles[i] += mutation_vector
                    particles[i] = np.clip(particles[i], lb, ub)

            # Linearly decrease inertia weight
            self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)  # Added line for inertia weight

        return self.global_best_position