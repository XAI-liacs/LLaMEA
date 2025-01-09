import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.9  # Initial inertia weight
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.de_scale_factor = 0.8
        self.de_crossover_rate = 0.9
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Particle Swarm Optimization step
            for i in range(self.pop_size):
                fitness = func(self.particles[i])
                evaluations += 1

                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i].copy()

                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            if evaluations < self.budget:
                # Adaptively update inertia weight
                self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)
                r1 = np.random.rand(self.pop_size, self.dim)
                r2 = np.random.rand(self.pop_size, self.dim)
                self.velocities = (
                    self.inertia_weight * self.velocities
                    + self.cognitive_const * r1 * (self.personal_best_positions - self.particles)
                    + self.social_const * r2 * (self.global_best_position - self.particles)
                )
                self.particles += self.velocities
                self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Differential Evolution step
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                mutant_vector = np.clip(x0 + self.de_scale_factor * (x1 - x2), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.de_crossover_rate, mutant_vector, self.particles[i])
                
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.personal_best_values[i]:
                    self.particles[i] = trial_vector
                    self.personal_best_values[i] = trial_fitness
                    self.personal_best_positions[i] = trial_vector

                if trial_fitness < self.global_best_value:
                    self.global_best_value = trial_fitness
                    self.global_best_position = trial_vector

                if evaluations >= self.budget:
                    break

        return self.global_best_position, self.global_best_value