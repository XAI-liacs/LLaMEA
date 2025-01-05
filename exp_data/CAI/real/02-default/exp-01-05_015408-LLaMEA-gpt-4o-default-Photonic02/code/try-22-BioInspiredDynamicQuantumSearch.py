import numpy as np

class BioInspiredDynamicQuantumSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim
        self.alpha = 0.5
        self.beta = 0.3
        self.mutation_rate = 0.05
        self.dynamic_factor = 0.1

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
                # Quantum inspired update with dynamic mutation
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

                # Dynamic mutation to escape local optima
                if np.random.rand() < self.mutation_rate:
                    quantum_population[i] = self.apply_dynamic_mutation(quantum_population[i])

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
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits):
        # Quantum rotation gate inspired update
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def apply_dynamic_mutation(self, quantum_bits):
        # Apply dynamic mutation based on a pre-defined factor
        mutation_strength = self.dynamic_factor * (np.random.rand(self.dim) - 0.5)
        mutated_bits = quantum_bits + mutation_strength
        mutated_bits = np.clip(mutated_bits, 0, 1)
        return mutated_bits