import numpy as np

class HybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.mutation_factor = 0.8
        self.best_pos = None
        self.best_val = np.inf

    def _initialize_particles(self):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_values

    def _update_velocity(self, velocity, position, personal_best_position, global_best_position):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best_position - position)
        social_velocity = self.social_coefficient * r2 * (global_best_position - position)
        new_velocity = self.inertia * velocity + cognitive_velocity + social_velocity
        return new_velocity

    def _mutate(self, position):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = indices
        mutant_vector = position[a] + self.mutation_factor * (position[b] - position[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_values = self._initialize_particles()
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                current_value = func(particles[i])
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                if current_value < self.best_val:
                    self.best_val = current_value
                    self.best_pos = particles[i]

                evaluations += 1
                if evaluations >= self.budget:
                    break

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], personal_best_positions[i], self.best_pos
                )
                particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Adaptive differential mutation
                if np.random.rand() < 0.1:  # With a small probability, mutate
                    particles[i] = self._mutate(particles)

        return self.best_pos, self.best_val