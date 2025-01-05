import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.mutation_rate = 0.1
        self.dynamic_population = True

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        quantum_population = np.random.rand(population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            if self.dynamic_population and evaluations > self.budget / 2:
                population_size = max(5, population_size // 2)

            for i in range(population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)

                if np.random.rand() < self.mutation_rate:
                    quantum_population[i] = self.mutate_quantum_bits(quantum_population[i])
                
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

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
        mutation_strength = np.random.randn(*quantum_bits.shape) * 0.05
        mutated_quantum_bits = quantum_bits + mutation_strength
        mutated_quantum_bits = np.clip(mutated_quantum_bits, 0, 1)
        return mutated_quantum_bits