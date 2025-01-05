import numpy as np

class QuantumInterferenceOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.interference_strength = 0.5
        self.mutation_rate = 0.1
        self.phase_shift = np.pi / 4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            new_quantum_population = self.apply_interference_pattern(quantum_population, best_index)

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    new_quantum_population[i] = self.mutate_quantum_bits(new_quantum_population[i])

                position_population[i] = self.quantum_to_position(new_quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    quantum_population[i] = new_quantum_population[i]

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def apply_interference_pattern(self, quantum_population, best_index):
        best_quantum_bits = quantum_population[best_index]
        interference_population = quantum_population + self.interference_strength * np.sin(2 * np.pi * (quantum_population - best_quantum_bits) + self.phase_shift)
        interference_population = np.clip(interference_population, 0, 1)
        return interference_population

    def mutate_quantum_bits(self, quantum_bits):
        mutation_vector = np.random.normal(0, 0.1, size=self.dim)
        mutated_bits = quantum_bits + mutation_vector
        mutated_bits = np.clip(mutated_bits, 0, 1)
        return mutated_bits