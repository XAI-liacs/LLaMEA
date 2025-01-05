import numpy as np

class AQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.w = 0.7  # Initial inertia weight
        self.c1 = 1.3  # Cognitive coefficient
        self.c2 = 1.7  # Social coefficient
        self.adaptation_factor = 0.1
        self.quantum_factor = 0.05
        self.bounds = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_superposition(self, position):
        alpha = np.random.rand(self.dim)
        superposed = alpha * self.global_best_position + (1 - alpha) * position
        noise = np.random.normal(0, self.quantum_factor, self.dim)
        new_position = superposed + noise
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def evolutionary_update(self):
        for i in range(self.population_size):
            if np.random.rand() < self.adaptation_factor:
                mutation = np.random.normal(0, 0.1, self.dim)
                self.population[i] += mutation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.population[i])
                evaluations += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.population[i].copy()

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.population[i].copy()

            self.evolutionary_update()

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.population[i] += self.velocities[i]

                # Quantum-inspired superposition state
                if np.random.rand() < self.adaptation_factor:
                    self.population[i] = self.quantum_superposition(self.population[i])

        return self.global_best_position, self.global_best_value