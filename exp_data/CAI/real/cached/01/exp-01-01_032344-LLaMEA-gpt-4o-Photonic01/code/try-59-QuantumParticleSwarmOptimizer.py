import numpy as np

class QuantumParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.9  # Dynamic inertia weight initialization
        self.cognitive_coef = 1.5  # Adjusted cognitive coefficient
        self.social_coef = 1.5
        self.quantum_coef = 0.1
        self.position = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf
        # Introduce dynamic inertia weight adjustment
        self.inertia_damp = 0.99

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

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coef * r1 * (self.personal_best_position[i] - self.position[i]) +
                                    self.social_coef * r2 * (self.global_best_position - self.position[i]) +
                                    self.quantum_coef * np.random.normal(0, 1, self.dim))
                self.position[i] += self.velocity[i]

                if np.random.rand() < 0.16:  # Apply Lévy flight with a slightly higher probability
                    self.position[i] += self.levy_flight(self.dim)

                self.position[i] = np.clip(self.position[i], lb, ub)

            self.inertia_weight *= self.inertia_damp  # Update inertia weight
            self.quantum_coef *= (0.98 if self.quantum_coef > 0.05 else 1)  # Update quantum coefficient

        return self.global_best_position, self.global_best_value