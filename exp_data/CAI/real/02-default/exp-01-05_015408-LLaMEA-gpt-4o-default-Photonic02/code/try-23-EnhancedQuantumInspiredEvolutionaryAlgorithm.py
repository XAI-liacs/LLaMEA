import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.alpha_min = 0.3
        self.alpha_max = 0.7
        self.beta = 0.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        quantum_population = np.random.rand(population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        best_fitness = fitness[best_index]
        stagnation_count = 0

        while evaluations < self.budget:
            for i in range(population_size):
                # Adaptive alpha based on stagnation
                alpha = self.adaptive_alpha(stagnation_count)

                # Quantum rotation gate: update quantum bits
                if np.random.rand() < alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                # Update best position
                if new_fitness < best_fitness:
                    best_index = i
                    best_position = position_population[i]
                    best_fitness = new_fitness
                    stagnation_count = 0  # reset stagnation counter
                else:
                    stagnation_count += 1

                if evaluations >= self.budget:
                    break

            # Dynamic population resizing based on stagnation
            if stagnation_count > 2 * population_size:
                population_size = max(5 * self.dim, population_size // 2)
                quantum_population, position_population, fitness = self.resize_population(quantum_population, position_population, fitness, population_size)
                stagnation_count = 0

        return best_position, best_fitness

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

    def adaptive_alpha(self, stagnation_count):
        # Adaptive alpha adjustment
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.tanh(stagnation_count / 10)

    def resize_population(self, quantum_population, position_population, fitness, new_size):
        # Resize populations and fitness arrays
        sorted_indices = np.argsort(fitness)
        quantum_population = quantum_population[sorted_indices[:new_size]]
        position_population = position_population[sorted_indices[:new_size]]
        fitness = fitness[sorted_indices[:new_size]]
        return quantum_population, position_population, fitness