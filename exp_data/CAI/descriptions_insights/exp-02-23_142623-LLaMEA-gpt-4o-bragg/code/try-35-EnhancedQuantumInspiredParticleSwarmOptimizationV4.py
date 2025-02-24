import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimizationV4:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * self.dim
        self.current_population_size = self.initial_population_size
        self.alpha = 0.75
        self.beta = 0.25
        self.best_global_position = None
        self.best_global_value = np.inf
        self.neighborhood_size = 5
        self.diversity_threshold = 0.1

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

        omega_start, omega_end = 0.9, 0.4
        while evaluations < self.budget:
            omega = omega_start - (omega_start - omega_end) * ((evaluations / self.budget) ** 2)
            self.neighborhood_size = max(3, int(0.15 * self.current_population_size))  
            for i in range(self.current_population_size):
                self.alpha = 0.5 + 0.5 * np.random.rand()
                self.beta = 0.5 - 0.5 * np.random.rand()
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
                    0.15 * (np.random.rand(self.dim) * (self.best_global_position - population[i])) +
                    0.05 * (np.random.rand(self.dim) * (np.mean(personal_best_positions, axis=0) - population[i]))
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

            diversity = np.std(population, axis=0).mean()
            mutation_prob = 0.1 * (1 - evaluations / self.budget) if diversity > self.diversity_threshold else 0.2
            for i in range(self.current_population_size):
                if np.random.rand() < mutation_prob:
                    mutation_step = np.random.normal(0, 1, self.dim)
                    mutated_position = np.clip(population[i] + mutation_step, lb, ub)
                    mutated_fitness = func(mutated_position)
                    if mutated_fitness < personal_best_values[i]:
                        personal_best_positions[i] = mutated_position
                        personal_best_values[i] = mutated_fitness

            elite_idx = np.argmin(personal_best_values)
            population[np.random.choice(self.current_population_size, 1)] = personal_best_positions[elite_idx]  

        return self.best_global_position, self.best_global_value