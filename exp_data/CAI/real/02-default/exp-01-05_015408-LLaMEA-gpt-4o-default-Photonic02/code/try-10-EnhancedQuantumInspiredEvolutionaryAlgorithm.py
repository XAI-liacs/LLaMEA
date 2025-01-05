import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7

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
                # Adaptive quantum rotation gate: update quantum bits
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index])

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Introduce crossover with a randomly selected partner
                if np.random.rand() < self.crossover_rate:
                    partner_index = np.random.randint(self.population_size)
                    position_population[i] = self.crossover(position_population[i], position_population[partner_index], lb, ub)

                # Apply mutation
                if np.random.rand() < self.mutation_rate:
                    position_population[i] = self.mutate(position_population[i], lb, ub)

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

    def crossover(self, individual, partner, lb, ub):
        # Perform uniform crossover between two individuals
        mask = np.random.rand(self.dim) > 0.5
        child = np.where(mask, individual, partner)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub):
        # Apply mutation by adding a small random value
        mutation_vector = np.random.normal(0, 0.1, self.dim)
        mutant = individual + mutation_vector
        return np.clip(mutant, lb, ub)