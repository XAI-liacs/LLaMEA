import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimizationV6:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.current_population_size = self.initial_population_size
        self.best_global_position = None
        self.best_global_value = np.inf
        self.neighborhood_size = 5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.current_population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.current_population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        evaluations = self.current_population_size

        if np.min(personal_best_values) < self.best_global_value:
            self.best_global_value = np.min(personal_best_values)
            self.best_global_position = population[np.argmin(personal_best_values)]

        while evaluations < self.budget:
            progress_ratio = evaluations / self.budget
            omega = 0.9 - 0.5 * ((progress_ratio) ** 0.5)  # Adaptive inertia weight
            dynamic_alpha = 0.5 + 0.5 * (1 - progress_ratio)  # Dynamic learning rate
            self.neighborhood_size = max(4, int(0.2 * self.current_population_size)) 
            for i in range(self.current_population_size):
                local_best_value = np.inf
                local_best_position = None
                neighbors = np.random.choice(self.current_population_size, self.neighborhood_size, replace=False)
                for neighbor in neighbors:
                    if personal_best_values[neighbor] < local_best_value:
                        local_best_value = personal_best_values[neighbor]
                        local_best_position = personal_best_positions[neighbor]
                velocity[i] = (
                    omega * velocity[i] +
                    dynamic_alpha * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                    (0.3 - progress_ratio) * np.random.rand(self.dim) * (local_best_position - population[i]) +
                    0.2 * (np.random.rand(self.dim) * (self.best_global_position - population[i]))
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

            if evaluations % (self.budget // 5) == 0:
                self.current_population_size = max(5, self.current_population_size // 2)
                population = population[:self.current_population_size]
                velocity = velocity[:self.current_population_size]
                personal_best_positions = personal_best_positions[:self.current_population_size]
                personal_best_values = personal_best_values[:self.current_population_size]

            mutation_prob = 0.1 * (1 - progress_ratio)
            for i in range(self.current_population_size):
                if np.random.rand() < mutation_prob:
                    mutation_step = np.random.normal(0, 0.1, self.dim)
                    mutated_position = np.clip(population[i] + mutation_step, lb, ub)
                    mutated_fitness = func(mutated_position)
                    if mutated_fitness < personal_best_values[i]:
                        personal_best_positions[i] = mutated_position
                        personal_best_values[i] = mutated_fitness

            elite_idx = np.argmin(personal_best_values)
            population[np.random.choice(self.current_population_size, 1)] = personal_best_positions[elite_idx]

        return self.best_global_position, self.best_global_value