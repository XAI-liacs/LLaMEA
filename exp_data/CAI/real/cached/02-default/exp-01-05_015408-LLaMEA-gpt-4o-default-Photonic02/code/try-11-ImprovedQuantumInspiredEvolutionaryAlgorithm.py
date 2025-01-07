import numpy as np

class ImprovedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.population_reduction_factor = 0.9

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

        while evaluations < self.budget:
            for i in range(population_size):
                # Adaptive quantum rotation gate: update quantum bits
                adaptive_beta = self.beta * (1 - evaluations / self.budget)
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], adaptive_beta)

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

                if evaluations >= self.budget:
                    break

            # Dynamic population size adjustment
            if evaluations < self.budget:
                population_size = max(2, int(population_size * self.population_reduction_factor))
                quantum_population = quantum_population[:population_size]
                position_population = position_population[:population_size]
                fitness = fitness[:population_size]

        return best_position, best_fitness

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, beta):
        # Quantum rotation gate inspired update with adaptive beta
        delta_theta = beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits