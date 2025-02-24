import numpy as np

class EnhancedQuantumInspiredParticleSwarmOptimization:
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
        self.evolutionary_step_size = 0.1  # New step size for evolutionary strategy

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

        omega_start, omega_end = 0.85, 0.35  # Adjusted inertia weight range
        while evaluations < self.budget:
            omega = omega_start - (omega_start - omega_end) * (evaluations / self.budget)
            for i in range(self.current_population_size):
                self.alpha = 0.6 + 0.4 * np.random.rand()  # Adjusted alpha range
                self.beta = 0.4 - 0.4 * np.random.rand()  # Adjusted beta range
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

            # Dynamic population resizing
            if evaluations % (self.budget // 5) == 0:
                new_population_size = max(3, self.current_population_size // 2)  # More aggressive resizing
                if new_population_size < self.current_population_size:
                    self.current_population_size = new_population_size
                    population = population[:self.current_population_size]
                    velocity = velocity[:self.current_population_size]
                    personal_best_positions = personal_best_positions[:self.current_population_size]
                    personal_best_values = personal_best_values[:self.current_population_size]

            # Guided Evolutionary Strategy
            for i in range(self.current_population_size):
                if np.random.rand() < 0.2 * (1 - evaluations / self.budget):  # Adjusted mutation probability
                    new_position = personal_best_positions[i] + self.evolutionary_step_size * np.random.normal(0, 1, self.dim)
                    new_position = np.clip(new_position, lb, ub)
                    new_fitness = func(new_position)
                    if new_fitness < personal_best_values[i]:
                        personal_best_positions[i] = new_position
                        personal_best_values[i] = new_fitness

            elite_idx = np.argmin(personal_best_values)
            population[np.random.choice(self.current_population_size)] = personal_best_positions[elite_idx]

        return self.best_global_position, self.best_global_value