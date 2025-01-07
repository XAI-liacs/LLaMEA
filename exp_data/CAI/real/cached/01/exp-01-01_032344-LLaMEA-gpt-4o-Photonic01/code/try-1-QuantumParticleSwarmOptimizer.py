import numpy as np

class QuantumParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7  # Changed from 0.5 to 0.7
        self.cognitive_coef = 2.0  # Changed from 1.5 to 2.0
        self.social_coef = 2.0  # Changed from 1.5 to 2.0
        self.quantum_coef = 0.1
        self.position = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf
        self.learning_rate_decay = 0.99  # New parameter
        self.boundary_buffer = 0.01  # New parameter

    def __call__(self, func):
        func_budget = self.budget
        num_evaluations = 0

        lb, ub = func.bounds.lb, func.bounds.ub
        self.position = lb + self.position * (ub - lb)

        while num_evaluations < func_budget:
            for i in range(self.population_size):
                if num_evaluations >= func_budget:
                    break
                fitness_value = func(self.position[i])
                num_evaluations += 1

                if fitness_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = fitness_value
                    self.personal_best_position[i] = self.position[i]

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = self.position[i]

            # Update particles' velocities and positions with adaptive learning rates
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coef * r1 * (self.personal_best_position[i] - self.position[i]) +
                                    self.social_coef * r2 * (self.global_best_position - self.position[i]) +
                                    self.quantum_coef * np.random.normal(0, 1, self.dim))
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], lb + self.boundary_buffer, ub - self.boundary_buffer)

            self.inertia_weight *= self.learning_rate_decay  # Apply decay
        return self.global_best_position, self.global_best_value