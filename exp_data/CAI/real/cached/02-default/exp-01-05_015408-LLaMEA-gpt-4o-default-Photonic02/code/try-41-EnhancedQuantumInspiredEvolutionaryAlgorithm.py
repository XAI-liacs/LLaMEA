import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.alpha = 0.5
        self.beta = 0.5
        self.dynamic_shrink_factor = 0.99  # To dynamically reduce population
        self.adaptive_rotation_factor = 0.1  # To adaptively adjust rotation based on improvement

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            previous_best_fitness = best_fitness
            for i in range(self.population_size):
                # Adaptive quantum rotation gate: update quantum bits
                if np.random.rand() < self.alpha:
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

                if evaluations >= self.budget:
                    break

            # Adaptive shrink of population size for faster convergence
            if best_fitness < previous_best_fitness:
                self.population_size = max(2, int(self.dynamic_shrink_factor * self.population_size))
                quantum_population = quantum_population[:self.population_size]
                position_population = position_population[:self.population_size]
                fitness = fitness[:self.population_size]

            # Adaptive control for quantum rotation angles
            self.beta *= (1 + self.adaptive_rotation_factor * np.sign(previous_best_fitness - best_fitness))

        return best_position, best_fitness

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits):
        # Adaptive Quantum rotation gate inspired update
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits