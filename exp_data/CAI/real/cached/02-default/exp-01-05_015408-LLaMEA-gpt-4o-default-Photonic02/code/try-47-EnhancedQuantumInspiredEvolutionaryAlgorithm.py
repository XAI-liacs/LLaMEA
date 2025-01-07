import numpy as np

class EnhancedQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_population_size = 10 * dim
        self.min_population_size = max(5, int(0.1 * dim))
        self.alpha = 0.5
        self.beta_initial = 0.5
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.max_population_size
        quantum_population = np.random.rand(population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        self.evaluations = population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while self.evaluations < self.budget:
            # Dynamically adjust the population size
            population_size = self.adjust_population_size(population_size)
            quantum_population, position_population, fitness = self.trim_population(quantum_population, position_population, fitness, population_size)

            for i in range(population_size):
                # Use dynamic quantum rotation influenced by the fitness landscape
                beta = self.beta_initial * (1 - self.evaluations / self.budget)
                
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], beta)

                # Convert quantum representation to classical position
                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)

                # Evaluate new position
                new_fitness = func(position_population[i])
                self.evaluations += 1

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                # Update best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if self.evaluations >= self.budget:
                    break

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        # Convert quantum bits to classical positions in the search space
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, beta):
        # Quantum rotation gate inspired update with dynamic beta
        delta_theta = beta * (best_quantum_bits - quantum_bits)
        new_quantum_bits = quantum_bits + delta_theta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def adjust_population_size(self, current_size):
        # Gradually reduce population size as evaluations progress
        reduction_rate = (self.max_population_size - self.min_population_size) / self.budget
        new_size = self.max_population_size - int(reduction_rate * self.evaluations)
        return max(self.min_population_size, new_size)

    def trim_population(self, quantum_pop, position_pop, fitness, new_size):
        # Trim population arrays to the new size
        quantum_pop = quantum_pop[:new_size]
        position_pop = position_pop[:new_size]
        fitness = fitness[:new_size]
        return quantum_pop, position_pop, fitness