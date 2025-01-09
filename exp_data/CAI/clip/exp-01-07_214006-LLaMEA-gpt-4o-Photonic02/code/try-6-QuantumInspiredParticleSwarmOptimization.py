import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 2.05  # Cognitive coefficient
        self.c2 = 2.05  # Social coefficient
        self.beta = 0.5  # Initial quantum position shift factor

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        velocities = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) / 20
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        self.current_evaluations += self.population_size

        while self.current_evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (self.current_evaluations / self.budget))
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                self.c2 * r2 * (global_best_position - population[i]))
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], bounds[:,0], bounds[:,1])

                current_fitness = func(population[i])
                self.current_evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[global_best_index]:
                        global_best_position = population[i]
                        global_best_index = i

            # Quantum tunneling strategy
            if self.current_evaluations < self.budget:
                for i in range(self.population_size):
                    self.beta = 0.5 * (1 - self.current_evaluations / self.budget)
                    quantum_shift = self.beta * (np.random.rand(self.dim) - 0.5)
                    quantum_position = global_best_position + quantum_shift
                    quantum_position = np.clip(quantum_position, bounds[:,0], bounds[:,1])
                    quantum_fitness = func(quantum_position)
                    self.current_evaluations += 1

                    if quantum_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = quantum_position
                        personal_best_fitness[i] = quantum_fitness
                        if quantum_fitness < personal_best_fitness[global_best_index]:
                            global_best_position = quantum_position
                            global_best_index = i

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]