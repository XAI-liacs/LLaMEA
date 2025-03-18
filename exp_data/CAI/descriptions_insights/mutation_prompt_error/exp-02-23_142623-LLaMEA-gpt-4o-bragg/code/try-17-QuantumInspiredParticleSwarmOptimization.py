import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.alpha = 0.75  # Exploration weight
        self.beta = 0.25   # Exploitation weight
        self.best_global_position = None
        self.best_global_value = np.inf
        self.neighborhood_size = 5  # Neighborhood size for local search

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
            for i in range(self.population_size):
                self.alpha = 0.5 + 0.5 * np.random.rand()
                self.beta = 0.5 - 0.5 * np.random.rand()
                local_best_value = np.inf
                local_best_position = None
                self.neighborhood_size = max(2, self.neighborhood_size - 1) if personal_best_values[i] < self.best_global_value else self.neighborhood_size + 1  # Change 1
                neighbors = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                for neighbor in neighbors:
                    if personal_best_values[neighbor] < local_best_value:
                        local_best_value = personal_best_values[neighbor]
                        local_best_position = personal_best_positions[neighbor]
                velocity[i] = (
                    omega * velocity[i] +
                    self.alpha * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                    self.beta * np.random.rand(self.dim) * (local_best_position - population[i])
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
            
            elite_idx = np.argmin(personal_best_values)  # New: Identify the elite
            population[np.random.choice(self.population_size)] = personal_best_positions[elite_idx]  # New: Introduce elitism

        return self.best_global_position, self.best_global_value