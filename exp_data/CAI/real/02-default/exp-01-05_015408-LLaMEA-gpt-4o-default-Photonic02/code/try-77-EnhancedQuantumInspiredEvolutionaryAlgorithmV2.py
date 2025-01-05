import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithmV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.6  # Increased for more exploration
        self.beta = 0.6  # Increased to adapt faster
        self.adaptive_factor = 0.15  # Increased for more responsive adaptation
        self.mutation_rate = 0.1  # New parameter for diversity enhancement

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive quantum rotation with mutation
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)
                if np.random.rand() < self.mutation_rate:
                    quantum_population[i] = self.mutate_quantum_bits(quantum_population[i])

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def mutate_quantum_bits(self, quantum_bits):
        mutation_vector = np.random.normal(0, 0.1, size=self.dim)
        new_quantum_bits = quantum_bits + mutation_vector
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits