import numpy as np

class QuantumParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9
        self.cognitive_coef = 1.5
        self.social_coef = 1.6
        self.quantum_coef = 0.1
        self.position = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf
        self.inertia_damp = 0.99
        self.memory = np.copy(self.position)
        self.memory_value = np.full(self.population_size, np.inf)

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

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
                    self.memory[i] = self.position[i]
                    self.memory_value[i] = fitness_value

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coef * r1 * (self.personal_best_position[i] - self.position[i]) +
                                    self.social_coef * r2 * (self.global_best_position - self.position[i]) +
                                    (self.quantum_coef + 0.05 * (self.global_best_value - self.personal_best_value[i])) * np.random.normal(0, 1, self.dim))
                self.position[i] += self.velocity[i]

                if np.random.rand() < 0.17:
                    self.position[i] += self.levy_flight(self.dim)

                # Apply opposition-based learning
                new_position = lb + ub - self.position[i]
                new_position = np.clip(new_position, lb, ub)
                if func(new_position) < fitness_value:
                    self.position[i] = new_position

                self.position[i] = np.clip(self.position[i], lb, ub)

            self.inertia_weight *= self.inertia_damp

        return self.global_best_position, self.global_best_value