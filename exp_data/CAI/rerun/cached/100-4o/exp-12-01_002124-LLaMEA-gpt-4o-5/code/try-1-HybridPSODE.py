import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.particles = np.random.uniform(-5, 5, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.global_best_position = None
        self.personal_best_values = np.full(self.pop_size, float('inf'))
        self.global_best_value = float('inf')
        self.f = 0.5  # DE scaling factor
        self.cr = 0.9  # DE crossover probability

    def __call__(self, func):
        evaluations = 0

        # Evaluate initial population
        values = np.apply_along_axis(func, 1, self.particles)
        evaluations += len(values)

        # Update personal and global bests
        for i in range(self.pop_size):
            if values[i] < self.personal_best_values[i]:
                self.personal_best_values[i] = values[i]
                self.personal_best_positions[i] = self.particles[i]
            if values[i] < self.global_best_value:
                self.global_best_value = values[i]
                self.global_best_position = self.particles[i]

        # Optimization loop
        while evaluations < self.budget:
            # Particle Swarm Optimization Update
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive = r1 * (self.personal_best_positions[i] - self.particles[i])
                social = r2 * (self.global_best_position - self.particles[i])
                self.velocities[i] = 0.5 * self.velocities[i] + cognitive + social
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5, 5)

            # Differential Evolution Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                trial_vector = np.copy(self.particles[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        trial_vector[j] = self.particles[a][j] + self.f * (self.particles[b][j] - self.particles[c][j])
                trial_vector = np.clip(trial_vector, -5, 5)
                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < values[i]:
                    self.particles[i] = trial_vector
                    values[i] = trial_value
                    if trial_value < self.personal_best_values[i]:
                        self.personal_best_values[i] = trial_value
                        self.personal_best_positions[i] = trial_vector
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial_vector
                if evaluations >= self.budget:
                    break
        return self.global_best_position