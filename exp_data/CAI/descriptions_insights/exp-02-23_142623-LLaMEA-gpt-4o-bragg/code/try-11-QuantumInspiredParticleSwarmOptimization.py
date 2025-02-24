import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.alpha = 0.75
        self.beta = 0.25
        self.gamma = 0.9
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

        omega_start, omega_end = 0.9, 0.4
        while evaluations < self.budget:
            omega = omega_start - (omega_start - omega_end) * (evaluations / self.budget)
            diversity = np.std(personal_best_positions, axis=0)  # New: Calculate diversity
            for i in range(self.population_size):
                learning_factor = np.clip(1 - diversity[i] / (ub - lb), 0.1, 0.9)  # New: Adaptive learning factor
                self.alpha = learning_factor * np.random.rand()
                self.beta = (1 - learning_factor) * np.random.rand()
                velocity[i] = (
                    omega * velocity[i] +
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