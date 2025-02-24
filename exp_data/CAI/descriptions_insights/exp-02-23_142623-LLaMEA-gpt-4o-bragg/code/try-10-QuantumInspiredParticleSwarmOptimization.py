import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.alpha = 0.75  # Exploration weight
        self.beta = 0.25   # Exploitation weight
        self.gamma = 0.9   # Inertia weight
        self.best_global_position = None
        self.best_global_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        if np.min(personal_best_values) < self.best_global_value:
            self.best_global_value = np.min(personal_best_values)
            self.best_global_position = population[np.argmin(personal_best_values)]

        omega_start, omega_end = 0.9, 0.4  # New: Dynamic inertia weight range
        while evaluations < self.budget:
            omega = omega_start - (omega_start - omega_end) * (evaluations / self.budget)  # New: Update inertia weight
            for i in range(self.population_size):
                self.alpha = 0.5 + 0.5 * np.random.rand()
                self.beta = 0.5 - 0.5 * np.random.rand()
                velocity[i] = (
                    omega * velocity[i] +  # Changed: from self.gamma to omega
                    self.alpha * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                    self.beta * np.random.rand(self.dim) * (self.best_global_position - population[i])
                )
                position = np.clip(population[i] + velocity[i], lb, ub)
                fitness = func(position)
                evaluations += 1

                if fitness < personal_best_values[i]:
                    personal_best_positions[i] = position
                    personal_best_values[i] = fitness

                if fitness < self.best_global_value:
                    self.best_global_value = fitness
                    self.best_global_position = position

                if evaluations >= self.budget:
                    break

        return self.best_global_position, self.best_global_value