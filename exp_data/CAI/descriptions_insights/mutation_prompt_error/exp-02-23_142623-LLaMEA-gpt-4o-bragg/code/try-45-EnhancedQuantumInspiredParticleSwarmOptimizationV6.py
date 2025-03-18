import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimizationV6:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 12 * self.dim  # Adjusted initial population size
        self.current_population_size = self.initial_population_size
        self.alpha = 0.65  # Adjusted alpha
        self.beta = 0.35  # Adjusted beta
        self.best_global_position = None
        self.best_global_value = np.inf
        self.dynamic_neighborhood_factor = 0.3  # New dynamic neighborhood factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.current_population_size, self.dim))
        velocity = np.random.uniform(-0.5, 0.5, (self.current_population_size, self.dim))  # Adjusted velocity range
        personal_best_positions = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        evaluations = self.current_population_size

        if np.min(personal_best_values) < self.best_global_value:
            self.best_global_value = np.min(personal_best_values)
            self.best_global_position = population[np.argmin(personal_best_values)]

        omega_start, omega_end = 0.85, 0.3  # Adjusted omega
        while evaluations < self.budget:
            omega = omega_start - (omega_start - omega_end) * ((evaluations / self.budget) ** 0.5)
            self.neighborhood_size = max(3, int(self.dynamic_neighborhood_factor * self.current_population_size))  # Dynamic neighborhood
            for i in range(self.current_population_size):
                self.alpha = 0.4 + 0.4 * np.random.rand()  # Adjusted alpha range
                self.beta = 0.2 + 0.2 * np.random.rand()  # Adjusted beta range
                local_best_value = np.inf
                local_best_position = None
                neighbors = np.random.choice(self.current_population_size, self.neighborhood_size, replace=False)
                for neighbor in neighbors:
                    if personal_best_values[neighbor] < local_best_value:
                        local_best_value = personal_best_values[neighbor]
                        local_best_position = personal_best_positions[neighbor]
                velocity[i] = (
                    omega * velocity[i] +
                    self.alpha * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                    self.beta * np.random.rand(self.dim) * (local_best_position - population[i]) +
                    0.2 * (np.random.rand(self.dim) * (self.best_global_position - population[i]))  # Adjusted
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

            if evaluations % (self.budget // 4) == 0:  # Adjusted frequency
                new_population_size = max(6, self.current_population_size // 2)
                if new_population_size < self.current_population_size:
                    self.current_population_size = new_population_size
                    population = population[:self.current_population_size]
                    velocity = velocity[:self.current_population_size]
                    personal_best_positions = personal_best_positions[:self.current_population_size]
                    personal_best_values = personal_best_values[:self.current_population_size]

            mutation_prob = 0.2 * (1 - evaluations / self.budget)  # Adjusted mutation probability
            for i in range(self.current_population_size):
                if np.random.rand() < mutation_prob:
                    mutation_step = np.random.normal(0, 0.15, self.dim)  # Adjusted mutation scale
                    mutated_position = np.clip(population[i] + mutation_step, lb, ub)
                    mutated_fitness = func(mutated_position)
                    if mutated_fitness < personal_best_values[i]:
                        personal_best_positions[i] = mutated_position
                        personal_best_values[i] = mutated_fitness

            elite_idx = np.argmin(personal_best_values)
            population[np.random.choice(self.current_population_size, 1)] = personal_best_positions[elite_idx]

        return self.best_global_position, self.best_global_value